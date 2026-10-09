"""OpenAI-compatible entrypoint for Interfaze Lite.

    POST /v1/chat/completions

One model name, any input. The brain decides which perception tools to run,
runs them, and answers. Structured tool output reaches the caller through the top-level
`precontext` field; prose goes in `choices[].message.content`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from . import forecasting
from . import guard as guardrails
from . import intent
from . import tools as toolkit
from .brain import STRUCTURED_MAX_TEMPERATURE, BrainClient, BrainError, ContextLengthError, Sampling
from .config import settings
from .contracts import InputFetchError
from .envelope import (
    Completion,
    PrecontextItem,
    Usage,
    new_request_id,
    sse,
)
from .filerefs import extract_from_messages
from .prompts import CUT_OFF_CALL as _CUT_OFF_CALL
from .prompts import NUDGE as _NUDGE
from .prompts import OUT_OF_STEPS as _OUT_OF_STEPS
from .prompts import SYSTEM_PROMPT
from .validate import (
    RequestError,
    check_api_key,
    extract_task,
    normalise_roles,
    resolve_task,
    strip_task_tags,
    validate_messages,
    validate_sampling,
)

log = logging.getLogger("interfaze.app")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

SCHEMA_PROMPT = """The user requires a structured answer that exactly fills the requested schema.
Fill each field from the context above, choosing the value by what the field asks for rather than
by what appears first. Names, numbers, codes and quoted text go in exactly as the context has them
and in full -- every part of a name, every digit and sign -- unless the user or the field asks
for a changed form, such as a translation. A value the context does not contain is null where the schema allows
null. Optional fields are filled too whenever the context has a value for them, and a list holds every item the
context has: an empty list means there were none. An item counts even when the context shows only part of it -- a
section or table that runs past the end of the page. An image shown beside a tool's reading is there for what the text cannot carry, such as
highlighting or colour. Do not add prose, explanations, or code fences around the result."""



@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = httpx.AsyncClient(follow_redirects=True)
    app.state.brain = BrainClient(app.state.http, settings)
    log.info("orchestrator ready; brain=%s", settings.brain_url)
    try:
        yield
    finally:
        await app.state.http.aclose()


app = FastAPI(title="interfaze-lite", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
async def health() -> dict[str, Any]:
    """Aggregate health. Reports each sidecar so one call tells you which is down,
    rather than having to infer it from a failing tool call."""
    out: dict[str, Any] = {
        "ok": True,
        "model": settings.model_name,
        "brain_ready": await app.state.brain.healthy(),
    }
    for name, base in (("perception", settings.perception_url), ("diarize", settings.diarize_url)):
        try:
            resp = await app.state.http.get(f"{base.rstrip('/')}/health", timeout=5)
            out[name] = resp.json() if resp.status_code < 400 else {"ok": False, "status": resp.status_code}
        except Exception as exc:
            out[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return out


@app.get("/v1/models")
async def models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {"id": settings.model_name, "object": "model", "owned_by": "interfaze"},
        ],
    }


async def _run_tool_loop(
    brain: BrainClient, messages: list[dict], ctx: toolkit.ToolContext, usage: Usage,
    extra_tools: list[dict] | None = None, sampling: Sampling | None = None,
    stop_after: str | None = None, tool_choice: str | dict = "auto",
    on_text: Callable[[str], None] | None = None,
    precontext: list[PrecontextItem] | None = None,
) -> tuple[list[dict], list[PrecontextItem], list[str], str, str]:
    """Select and run tools until the model stops asking for them.

    Returns the extended message list, the precontext to hand the caller, the names of
    the tools that ran, any answer this turn produced outright, and why that generation
    stopped. The last of those matters because the answer can be returned verbatim: a
    caller that capped max_tokens has to be told when the text it gets back was cut
    short. Capped at `max_tool_steps`; a model that keeps calling tools without
    converging would otherwise run until the request times out.
    """
    precontext = [] if precontext is None else precontext
    used: list[str] = []
    # "none" was applied by withholding the caller's functions; what reaches the model
    # is "auto" or a forced choice, and a forced choice binds only the first turn.
    choice = tool_choice if tool_choice not in ("auto", "none") else "auto"
    # The file tools only when there is a file: offered anyway, a text-only request got
    # an OCR call on a made-up reference. Translation and forecasting only when the
    # request asks for them (intent.py).
    series = (ctx.text_tools is None or "forecast" in ctx.text_tools) and bool(
        forecasting.dataset_from_text(ctx.prompt or ""))
    schemas = toolkit.schemas(files=bool(ctx.refs.refs), text=ctx.text_tools, series_in_prompt=series)

    # Caller-supplied functions sit alongside ours. We cannot execute them -- they run
    # on the caller's side -- so a call to one ends the loop and is handed back as
    # tool_calls, exactly as the OpenAI API does.
    theirs = {t["function"]["name"] for t in (extra_tools or [])
              if isinstance(t, dict) and t.get("type") == "function" and t.get("function")}
    schemas = schemas + list(extra_tools or [])

    nudged = False
    shown = False
    # Tool results the caller sent, from functions it ran -- a web search that came back
    # with twenty scraped pages -- count against the same window as ours.
    _trim_tool_messages(messages, ctx.settings.max_context_chars)
    # And a tool has run: answering from them is not answering without one.
    ran = any(m.get("role") == "tool" for m in messages)
    start = len(messages)
    # Calls of ours already answered -- in this request, or carried in from an earlier
    # one with the relay's results -- by what they asked. The same call again is answered
    # from that rather than run and shown again: through the relay, each round re-ran the
    # last round's OCR and detection, and one receipt was detected seven times over.
    answered = _answered_calls(messages)
    repeats = 0

    for step in range(ctx.settings.max_tool_steps):
        # Text streams live unless this turn may yet be discarded for the nudge below.
        live = on_text if (not ctx.refs.refs or used or ran or nudged) else None
        held = _Narration(live) if live else None
        reply = await brain.select_tools(messages, schemas, sampling, tool_choice=choice,
                                         on_text=held.feed if held else None)
        if held:
            held.end(called_tools=bool(reply.tool_calls))
        choice = "auto"
        usage.prompt_tokens += reply.prompt_tokens
        usage.completion_tokens += reply.completion_tokens
        usage.reasoning_tokens += reply.reasoning_tokens

        if not reply.tool_calls:
            # A file was attached, no tool has run, and the model is answering anyway.
            # It is a vision model, so it thinks it can just read the image -- and its
            # reading invents text: asked where a shop was, it produced "123 Main St,
            # New York, NY 10001" off a receipt that says nothing of the kind. Nothing
            # downstream can catch that, because a confident wrong answer looks exactly
            # like a right one. Give it one chance to reconsider with the omission
            # named; if it still declines, it presumably has a reason.
            if ctx.refs.refs and not used and not ran and not nudged:
                nudged = True
                log.info("step %d: answered with no tool despite %d attached file(s); "
                         "nudging", step, len(ctx.refs.refs))
                messages = [*messages, {"role": "user", "content": _NUDGE}]
                continue
            return messages, precontext, used, reply.content, reply.finish_reason

        if theirs and any(c.name in theirs for c in reply.tool_calls):
            # Our own calls and their results so far go back with the caller's, for the
            # caller to send again with its answers. Without them the next request has
            # lost them: through the relay, a receipt comparison read both receipts again
            # on each of five rounds.
            context = [m for m in messages[start:] if m.get("role") in ("assistant", "tool")]
            raise _CallerToolCalls(
                [c for c in reply.tool_calls if c.name in theirs], precontext, usage, context)

        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [c.as_dict() for c in reply.tool_calls],
            }
        )

        for call in reply.tool_calls:
            # Arguments carry the caller's content; a zero-data-retention caller's stay out
            # of the logs.
            log.info("step %d: %s(%s)", step, call.name,
                     "<redacted>" if ctx.zdr else call.arguments[:200])

        # A call already answered, or asked twice in this turn, is not run again.
        keys = [toolkit.memo_key(call.name, call.arguments) for call in reply.tool_calls]
        repeat = [k is not None and (k in answered or k in keys[:i]) for i, k in enumerate(keys)]
        if all(repeat):
            repeats += 1
            if repeats >= 2:
                # Asking again for what it already has, twice over: it will not converge.
                return ([*messages, {"role": "user", "content": _OUT_OF_STEPS}],
                        precontext, used, "", "stop")
        else:
            repeats = 0

        if reply.finish_reason == "length":
            # Cut off mid-call: run as they stand, the arguments would be read as whatever
            # the parser could salvage.
            results = [toolkit.ToolResult(model_facing={"error": _CUT_OFF_CALL})
                       for _ in reply.tool_calls]
        else:
            # The model can ask for several tools at once -- read this PDF and find the
            # figures in it -- and they do not depend on each other, so running them one
            # after another just adds their latencies together.
            fresh = await asyncio.gather(*(
                toolkit.dispatch(call.name, call.arguments, ctx)
                for call, again in zip(reply.tool_calls, repeat, strict=True) if not again
            ))
            ran_now = iter(fresh)
            results = [_ALREADY_ANSWERED if again else next(ran_now) for again in repeat]

        for call, key, again, result in zip(reply.tool_calls, keys, repeat, results, strict=True):
            content = _as_json(result.model_facing)
            if again:
                content = _ALREADY_ANSWERED_NOTE + answered.get(key, "")
            else:
                used.append(call.name)
                # The same result again -- one OCR read asked for markdown, then for line
                # boxes -- is shown once: the playground drew two identical OCR cards.
                if not any(p.name == call.name and p.result == result.precontext for p in precontext):
                    precontext.append(PrecontextItem(name=call.name, result=result.precontext))
                if key is not None and "error" not in result.model_facing:
                    answered[key] = content
            messages.append({"role": "tool", "tool_call_id": call.id, "content": content})

        _trim_tool_messages(messages, ctx.settings.max_context_chars)
        # Only once a tool has actually read the file. Shown after a failed OCR call, the
        # model read the receipt by eye and answered 144.68 for a total of 144.02.
        if not shown and any("error" not in result.model_facing for result in results):
            messages, shown = await _show_images(brain, messages, ctx), True

        # A routed task answers with the tool's own output, so once that tool has
        # succeeded another turn is a full generation nobody reads -- measured at ~20 s
        # on a transcript the model restated before being discarded. A failed call
        # still goes back to the model to correct.
        if stop_after and any(call.name == stop_after and "error" not in result.model_facing
                              for call, result in zip(reply.tool_calls, results, strict=True)):
            return messages, precontext, used, "", "stop"

    return [*messages, {"role": "user", "content": _OUT_OF_STEPS}], precontext, used, "", "stop"


_IMAGE_MENTION = re.compile(r"^\[image (ref-\d+)\]$|^\[file (ref-\d+): .*\]$")
_MAX_SHOWN_IMAGES = 4


async def _show_images(brain: BrainClient, messages: list[dict],
                       ctx: toolkit.ToolContext) -> list[dict]:
    """The caller's images, put back beside their references once a tool has read them.

    Withheld until then on purpose: shown an image first, the model transcribes it by
    eye instead of calling a tool, and invents text. But with the reading in hand, the
    image carries what the reading cannot -- the highlighted line on a receipt, a
    colour, a crossed-out item -- and a receipt's "which item is highlighted" was
    unanswerable without it. Bounded like grounding input; one that cannot be fetched
    stays a reference.
    """
    budget = ctx.settings.answer_image_max_pixels
    if budget <= 0:
        return messages
    shown, out = 0, []
    for message in messages:
        content = message.get("content")
        if message.get("role") != "user" or not isinstance(content, list):
            out.append(message)
            continue
        parts: list[dict] = []
        for part in content:
            parts.append(part)
            found = _IMAGE_MENTION.match(part.get("text", "")) if part.get("type") == "text" else None
            ref = ctx.refs.refs.get(next((g for g in found.groups() if g), "")) if found else None
            if not ref or shown >= _MAX_SHOWN_IMAGES:
                continue
            if not (part["text"].startswith("[image") or ref.mime.startswith("image/")):
                continue
            try:
                url, size = await brain.viewable(ref.url, budget, inline=True)
            except Exception as exc:
                log.info("not showing %s: %s", ref.ref_id, exc)
                continue
            if size is None:
                # Did not decode as an image. A PDF sent as an image part lands here, and
                # passed on, vLLM refused the whole request.
                continue
            parts.append({"type": "image_url", "image_url": {"url": url}})
            shown += 1
        out.append({**message, "content": parts})
    return out


_SOMETHING_WENT_WRONG = "Something went wrong. Please try again."

_ALREADY_ANSWERED_NOTE = ("This exact call was already made, and its result is below. "
                          "Answer from it; do not call it again.\n")
_ALREADY_ANSWERED = toolkit.ToolResult(model_facing={"already_answered": True})


def _answered_calls(messages: list[dict]) -> dict[str, str]:
    """Our calls in the conversation that have a result, by memo key, with that result."""
    asked: dict[str, str] = {}
    out: dict[str, str] = {}
    for message in messages:
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                fn = call.get("function") or {}
                key = toolkit.memo_key(fn.get("name"), fn.get("arguments"))
                if key is not None and call.get("id"):
                    asked[call["id"]] = key
        elif message.get("role") == "tool" and message.get("tool_call_id") in asked:
            out[asked[message["tool_call_id"]]] = message.get("content") or ""
    return out


class _Narration:
    """A streamed turn's text, held until it is plainly the answer.

    Before calling a tool the model sometimes says what it is about to do -- "Let me
    read the surrounding text to identify which one is the architecture diagram" -- and
    streamed live, that line opened the answer on the landing, though the buffered
    response never shows it. So a turn's text waits until it outgrows a sentence or two,
    or the turn ends; a turn that ends in a tool call drops what it held.
    """

    # About 75 tokens: half a second before a real answer starts to stream.
    LIMIT = 300

    def __init__(self, emit: Callable[[str], None]):
        self.emit = emit
        self.held: list[str] = []
        self.size = 0
        self.open = False

    def feed(self, text: str) -> None:
        if self.open:
            self.emit(text)
            return
        self.held.append(text)
        self.size += len(text)
        if self.size > self.LIMIT:
            self._release()

    def end(self, *, called_tools: bool) -> None:
        if called_tools and not self.open:
            self.held = []
        else:
            self._release()

    def _release(self) -> None:
        self.open = True
        text, self.held = "".join(self.held), []
        if text:
            self.emit(text)


class _CallerToolCalls(Exception):
    """The model asked for a function the caller owns. Hand it back rather than run it."""

    def __init__(self, calls, precontext, usage, context=()):
        super().__init__("caller tool calls")
        self.calls = calls
        self.precontext = precontext
        self.usage = usage
        # This request's own tool calls and results, as messages: `interfaze_context`.
        self.context = list(context)


def _trim_tool_messages(messages: list[dict], budget: int) -> None:
    """Keep the running conversation inside the model's context window.

    Each tool result is capped on its own, but the loop can run several of them and it
    is the *sum* that has to fit. Without this a two-document request overflows on the
    synthesis call, which fails the whole request after all the expensive work is done.
    Oldest results are truncated first: the most recent tool output is the one the
    model is most likely to be answering from.
    """
    indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    total = sum(len(messages[i].get("content") or "") for i in indices)
    if total <= budget:
        return
    for i in indices:
        if total <= budget:
            break
        content = messages[i].get("content") or ""
        keep = max(2000, len(content) - (total - budget))
        if keep >= len(content):
            continue
        messages[i] = {**messages[i],
                       "content": content[:keep] + "\n...[truncated]"}
        total -= len(content) - keep




# interfaze's `imageUrlsInLatestPrompt`: by extension only, as it notes.
_IMAGE_URL = re.compile(r"https?://[^\s]+\.(?:jpg|jpeg|png)", re.I)
# interfaze's stand-in prompt when the latest message carries files and no text.
_FILES_ONLY_PROMPT = ("Analyze the uploaded files and automatically select the most suitable "
                      "tools based on file types and any provided schema requirements.")


def _latest_prompt(messages: list[dict], refs) -> tuple[str, list[str]]:
    """The latest user message's text, and the images it carries, for the guard."""
    latest = next((m for m in reversed(messages) if m.get("role") == "user"), None)
    if latest is None:
        return "", []
    content = latest.get("content")
    parts = [{"type": "text", "text": content}] if isinstance(content, str) else (content or [])
    texts: list[str] = []
    images: list[str] = []
    for part in parts:
        if not isinstance(part, dict) or part.get("type") != "text":
            continue
        found = _IMAGE_MENTION.match(part.get("text", ""))
        if found:
            ref = refs.refs.get(next((g for g in found.groups() if g), ""))
            if ref and (part["text"].startswith("[image") or ref.mime.startswith("image/")):
                images.append(ref.url)
            continue
        if part.get("text", "").startswith("[audio ref-"):
            continue
        texts.append(part["text"])
    text = " ".join(t for t in texts if t.strip()).strip()
    for url in _IMAGE_URL.findall(text):
        if url not in images:
            images.append(url)
    if not text and (images or refs.refs):
        text = _FILES_ONLY_PROMPT
    return text, images


async def _run_guard(codes: list[str], messages: list[dict], refs, brain: BrainClient,
                     usage: Usage) -> tuple[bool, str, list[PrecontextItem]]:
    """The guard's verdict on the latest user message: (safe, blocked content, precontext)."""
    text, image_urls = _latest_prompt(messages, refs)

    checks = guardrails.wants_images(codes)
    image_jobs = [brain.image_safety(url) for url in image_urls] if any(checks.values()) else []
    verdict, *scores = await asyncio.gather(brain.text_safety(text), *image_jobs)

    usage.prompt_tokens += int(verdict.get("prompt_tokens", 0))
    usage.completion_tokens += int(verdict.get("completion_tokens", 0))
    for s in scores:
        usage.prompt_tokens += int(s.get("prompt_tokens", 0))
        usage.completion_tokens += int(s.get("completion_tokens", 0))
    images = [guardrails.image_result(s["adult"], s["racy"], s["gore"]) for s in scores]

    safe, content, items = guardrails.verdict(verdict.get("output", ""), codes, images or None)
    return safe, content, [PrecontextItem(name=i["name"], result=i["result"]) for i in items]


def _system_texts(content: Any) -> list[str]:
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        return [p["text"] for p in content
                if isinstance(p, dict) and p.get("type") == "text" and isinstance(p.get("text"), str)]
    return []


def _as_json(value: Any) -> str:
    # Characters as they are, as JSON.stringify writes them. Escaped, a task result that
    # a client shows as text read "\u4f60\u597d" where beta's read "你好".
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False, default=str)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    request_id = new_request_id()
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400,
                            content=_error("request body is not valid JSON",
                                           "invalid_request_error", request_id))

    try:
        check_api_key(request.headers.get("authorization"), settings.api_key,
                      request.headers.get("x-api-admin-key"), settings.admin_key)
        raw_messages = body.get("messages")
        validate_messages(raw_messages)
        # Before the task tag is read or the caller's system prompt is collected --
        # both find their input by testing for the `system` role.
        raw_messages = normalise_roles(raw_messages)
        validate_sampling(body, max_output_tokens=settings.max_output_tokens)
        task = extract_task(raw_messages)
        raw_messages = strip_task_tags(raw_messages)
        # `<guard>` is a directive to this service, as `<task>` is, not an instruction for
        # the model; the SDK's `guard=[...]` lowers to it.
        guard_codes, raw_messages = guardrails.extract_from_messages(raw_messages)
    except RequestError as exc:
        return JSONResponse(status_code=exc.status,
                            content=_error(str(exc), "invalid_request_error", request_id))

    want_stream = bool(body.get("stream", False))
    show_debug = request.headers.get("x-show-additional-info") == "true"
    schema = (body.get("response_format") or {}).get("json_schema")
    # The SDK sends an empty schema alongside `task=` to signal "raw output". Guided
    # decoding against {} constrains nothing and vLLM rejects it, so treat it as absent.
    if schema and not (schema.get("schema") or {}):
        schema = None
    # json_object is guided decoding too, against any object. Answered as prose, the
    # model wrapped its JSON in a markdown fence and the reply did not parse.
    if not schema and (body.get("response_format") or {}).get("type") == "json_object":
        schema = {"name": "json_object", "schema": {"type": "object"}}
    # As interfaze does (utils/messaging.ts): a task returns raw tool output and a schema
    # demands structured JSON, so asking for both is refused, not half-honoured. Lite
    # used to answer with the task body and ignore the schema, which a caller's SDK
    # then failed to parse as a NoObjectGeneratedError far from the cause.
    if task and schema and (schema.get("schema") or {}).get("properties"):
        return JSONResponse(status_code=400, content=_error(
            "Non-empty schema is not allowed to be used with run tasks",
            "invalid_request_error", request_id))
    caller_tools = [t for t in (body.get("tools") or [])
                    if isinstance(t, dict) and t.get("type") == "function"]
    # The caller's tool_choice, which was replaced by "auto": "none" returned a call
    # anyway. "none" withholds their functions; "required" or a named function forces
    # the first turn.
    tool_choice = body.get("tool_choice") or "auto"
    if tool_choice == "none":
        caller_tools = []
    effort = str(body.get("reasoning_effort") or "").lower()
    think = effort in ("low", "medium", "high", "xhigh")
    # What the caller asked for, kept separate from this service's defaults. Ignoring
    # these meant max_tokens never capped anything and finish_reason was always "stop".
    sampling = Sampling(
        max_tokens=body.get("max_completion_tokens") or body.get("max_tokens"),
        temperature=body.get("temperature"),
        top_p=body.get("top_p"),
    )

    # Off the loop: this base64-decodes every attachment and writes it to disk, which
    # on a multi-megabyte upload is long enough to stall the other requests sharing
    # this worker. It runs on every request that carries a file.
    refs, messages = await asyncio.to_thread(extract_from_messages, raw_messages)
    manifest = refs.manifest()

    # One system message, always. the brain's chat template rejects a second one with
    # "System message must be at the beginning", so the file manifest is appended to
    # the prompt rather than sent as its own turn.
    # Text parts count too: a system prompt sent as an array of parts was dropped.
    caller_system = "\n\n".join(
        text for m in messages if m.get("role") == "system"
        for text in _system_texts(m.get("content")) if text.strip()
    )
    rest = [m for m in messages if m.get("role") != "system"]
    # Restated beside the latest user turn as well. In the system prompt alone it sat
    # behind the tool instructions and lost: told to reply with only a code, the model
    # answered "Your test code is TULIP-7429."; restated here it answered the code, on
    # every probe. Skipped for long system prompts, where repeating them costs more
    # than it buys.
    if caller_system and len(caller_system) <= 2000:
        rest = _remind(rest, caller_system)

    parts = [SYSTEM_PROMPT]
    if manifest:
        parts.append(manifest)
    if caller_system:
        parts.append(caller_system)
    if task:
        routed = resolve_task(task)
        parts.append(
            f"The caller has explicitly routed this request to the `{routed}` tool. "
            f"Call `{routed}` on the supplied input. Do not call any other tool.")
    # Last, so what comes before it is the same on every request and stays cached; a
    # date, not a time, so it changes once a day. Without it "a 5-year overview" ended
    # on whatever year the model last saw.
    parts.append(f"Today's date: {datetime.now(timezone.utc).date().isoformat()}")

    convo: list[dict] = [{"role": "system", "content": "\n\n".join(parts)}]
    convo.extend(rest)

    # Whether _stream_response has taken responsibility for deleting the materialised
    # uploads. Keying that off `want_stream` leaked them: a request can ask for a
    # stream and still be answered with JSON -- structured output is routed away from
    # streaming deliberately, and a caller-owned tool call returns early -- and in
    # those cases the generator that does the cleanup never runs.
    streaming_owns_refs = False

    brain: BrainClient = app.state.brain
    usage = Usage()
    ctx = toolkit.ToolContext(refs=refs, http=app.state.http, settings=settings,
                              ground=brain.ground, structured=brain.structured, usage=usage,
                              wants_geometry=_schema_wants_geometry(schema),
                              zdr=request.headers.get("x-interfaze-zdr") == "true")

    # Translation and forecasting are offered when the request's instruction asks for
    # them, or when it is routed to one of them. Offered on every request, they were
    # called where nobody asked -- an English answer "translated" into English.
    latest_text, _ = _latest_prompt(messages, refs)
    ctx.prompt = latest_text
    ctx.text_tools = tuple(
        name for name, wanted in (("translate", intent.wants_translation(latest_text)),
                                  ("forecast", intent.wants_forecast(latest_text)))
        if wanted or (task and resolve_task(task) == name))

    # Guardrails run first, as in interfaze: a request that fails them is answered with
    # the verdict and never reaches a tool or the model.
    guard_items: list[PrecontextItem] = []
    if guard_codes is not None:
        try:
            safe, verdict_text, guard_items = await _run_guard(guard_codes, messages, refs, brain, usage)
        except InputFetchError:
            refs.cleanup()
            return JSONResponse(status_code=400, content=_error(
                "Failed to download image from the provided url", "invalid_request_error", request_id))
        except Exception:
            log.exception("guard failed")
            refs.cleanup()
            return JSONResponse(status_code=500, content=_error(
                "Failed to perform content safety check", "server_error", request_id))
        if not safe:
            blocked = Completion(request_id=request_id, model=settings.model_name, usage=usage,
                                 precontext=list(guard_items))
            if want_stream:
                return StreamingResponse(
                    _stream_text(blocked, verdict_text, refs, show_debug),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )
            refs.cleanup()
            return JSONResponse(blocked.message(verdict_text))

    # A streamed answer streams its tool turns: the text reaches the caller as it is
    # generated. Structured output and routed tasks deliver a finished object, and a
    # reasoning request synthesises separately with thinking on, so those three keep
    # the buffered loop below.
    if want_stream and not schema and not task and not think:
        streaming_owns_refs = True
        completion = Completion(
            request_id=request_id, model=settings.model_name, usage=usage,
            precontext=list(guard_items),
            debug={"internal_tool_used": [], "preConfig": {"tools": []}} if show_debug else None)
        return StreamingResponse(
            _stream_turns(brain, convo, ctx, completion, caller_tools, sampling,
                          tool_choice, refs, show_debug),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        try:
            # A structured answer's tool turns run calm too: their text is discarded, and at the
            # playground's temperature of 1 they varied which tools ran with what arguments --
            # bounds asked for, the file named by URL, ocr called twice -- and the schema
            # filled from each a little differently. The whole request at 0.3 filled a
            # paper's section list on 8 runs of 8.
            loop_sampling = (replace(sampling, temperature=STRUCTURED_MAX_TEMPERATURE)
                             if schema and (sampling.temperature or 0) > STRUCTURED_MAX_TEMPERATURE
                             else sampling)
            convo, precontext, used, direct, direct_reason = await _run_tool_loop(
                brain, convo, ctx, usage, caller_tools, loop_sampling,
                stop_after=resolve_task(task) if task else None, tool_choice=tool_choice,
                precontext=list(guard_items))
        except _CallerToolCalls as handoff:
            handed = Completion(
                request_id=request_id, model=settings.model_name, usage=handoff.usage,
                precontext=handoff.precontext,
            )
            if want_stream:
                streaming_owns_refs = True
                return StreamingResponse(
                    _stream_tool_calls(handed, handoff.calls, refs, show_debug, handoff.context),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )
            return JSONResponse(handed.tool_calls(handoff.calls, handoff.context))

        completion = Completion(
            request_id=request_id,
            model=settings.model_name,
            usage=usage,
            precontext=precontext,
            debug={"internal_tool_used": used, "preConfig": {"tools": used}} if show_debug else None,
        )

        # A routed <task> is a request for the tool's own output, not a prose answer
        # about it. Returning the raw precontext entry keeps `client.tasks.ocr(...)`
        # returning structured data instead of a paragraph describing it.
        if task and precontext:
            routed = resolve_task(task)
            items = [p for p in precontext if p.name == routed] or precontext[:1]
            # Echoed under the name the caller routed, not the tool's own, so a request
            # for `speech_to_text` is not answered by something calling itself `stt`.
            named = {**items[0].as_dict(), "name": task}
            if len(items) > 1:
                # Every call, as interfaze answers a task that ran its tool more than once:
                # a translation into three languages came back as the first one alone.
                named["results"] = [{**i.as_dict(), "name": task} for i in items]
            routed_body = _as_json(named)
            # Streams too. A routed task is still a completion, and the caller chose
            # the transport: answering a `stream: true` request with a JSON body left
            # the playground -- which always streams -- reading an empty response.
            if want_stream:
                streaming_owns_refs = True
                # With a response_format interfaze streams the result alone, without the
                # inline <precontext> copy of it. The playground's run-task cards always
                # send one, and drew lite's answer differently from beta's.
                return StreamingResponse(
                    _stream_text(completion, routed_body, refs, show_debug and not schema),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )
            return JSONResponse(completion.message(routed_body))

        # Guided decoding needs the whole generation to constrain, so the object is
        # built in one shot. That is a generation constraint, not a transport one: a
        # caller that asked for a stream still gets one, carrying the finished object.
        # Returning JSON to a stream request instead left OpenAI-compatible clients
        # reading zero chunks.
        if schema:
            # Its own system prompt, not the tool-calling one. That prompt tells the model
            # to call tools and reason freely, which guided decoding forbids; with both in
            # context a 28-item receipt came back as one item and '}, {' in every string,
            # or as whitespace to the token cap. The tool results are already in `convo`,
            # and the caller's own instructions still apply.
            system = "\n\n".join(p for p in (SCHEMA_PROMPT, caller_system) if p)
            body = convo[1:] if ctx.wants_geometry else _without_page_geometry(convo[1:])
            shaped = [{"role": "system", "content": system}, *body]
            reply = await brain.structured(shaped, schema, sampling)
            usage.prompt_tokens += reply.prompt_tokens
            usage.completion_tokens += reply.completion_tokens
            usage.reasoning_tokens += reply.reasoning_tokens
            if reply.finish_reason != "length" and not _parses(reply.content):
                # Guided decoding still lost its place once in ten on a dense schema -- a key
                # with no colon after it. One more draw; kept only if it parses.
                log.warning("structured reply did not parse; drawing again")
                again = await brain.structured(
                    shaped, schema, replace(sampling, temperature=STRUCTURED_MAX_TEMPERATURE))
                usage.prompt_tokens += again.prompt_tokens
                usage.completion_tokens += again.completion_tokens
                usage.reasoning_tokens += again.reasoning_tokens
                if _parses(again.content):
                    reply = again
            if want_stream:
                streaming_owns_refs = True
                return StreamingResponse(
                    _stream_text(completion, reply.content, refs, show_debug,
                                 reply.finish_reason),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )
            return JSONResponse(completion.message(reply.content, reply.finish_reason))

        # The selection turn often answers outright, and that answer is complete: by
        # the time it runs, the tool results are already in its context. Regenerating
        # it is a second full generation of text the caller never sees -- measured at
        # about a third of the completion tokens on a tool-using request, and a second
        # of wall clock.
        #
        # This used to be skipped whenever a tool had run, which switched the saving
        # off in exactly the case that pays for it and left it on where there was no
        # duplicate to avoid. It is still skipped when the caller asked to reason: the
        # selection turn runs with thinking off, so reusing it would quietly drop the
        # reasoning that was asked for.
        if settings.reuse_tool_turn_answer and direct.strip() and not think:
            return JSONResponse(completion.message(direct, direct_reason))

        if want_stream:
            streaming_owns_refs = True
            return StreamingResponse(
                _stream_response(brain, convo, completion, refs, think, effort,
                                 sampling, show_debug),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        reply = await brain.synthesise(convo, think=think, effort=effort,
                                       sampling=sampling)
        usage.prompt_tokens += reply.prompt_tokens
        usage.completion_tokens += reply.completion_tokens
        usage.reasoning_tokens += reply.reasoning_tokens
        completion.reasoning = reply.reasoning or None
        return JSONResponse(
            completion.message(reply.content or direct, reply.finish_reason))

    except ContextLengthError as exc:
        return JSONResponse(status_code=400, content=_error(str(exc), "invalid_request_error", request_id))
    except BrainError as exc:
        log.exception("brain failure")
        return JSONResponse(status_code=502, content=_error(_SOMETHING_WENT_WRONG, "brain_error", request_id))
    except Exception:
        # Logged in full; the caller gets interfaze's words, not our traceback.
        log.exception("request failed")
        return JSONResponse(status_code=500, content=_error(_SOMETHING_WENT_WRONG, "internal_error", request_id))
    finally:
        if not streaming_owns_refs:
            refs.cleanup()


def _remind(messages: list[dict], instructions: str) -> list[dict]:
    """The caller's system instructions, restated at the end of the latest user turn."""
    note = f"(Instructions from the caller, which still apply: {instructions})"
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") != "user":
            continue
        content = messages[i].get("content")
        if isinstance(content, list):
            content = [*content, {"type": "text", "text": note}]
        else:
            content = f"{content or ''}\n\n{note}"
        return [*messages[:i], {**messages[i], "content": content}, *messages[i + 1:]]
    return messages


def _role_chunk(completion: Completion) -> dict:
    """The opening chunk. Some clients use it to open the message before text arrives."""
    return {
        "id": completion.request_id,
        "object": "chat.completion.chunk",
        "created": completion.created,
        "model": completion.model,
        "choices": [
            {"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}
        ],
    }


def _final_chunk(completion: Completion, finish_reason: str) -> dict:
    return {
        "id": completion.request_id,
        "object": "chat.completion.chunk",
        "created": completion.created,
        "model": completion.model,
        "usage": completion.usage.as_dict(),
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
    }


def _with_context(chunk: dict, context) -> dict:
    """A tool-call turn's last chunk, carrying this request's own exchange when it had one."""
    return {**chunk, "interfaze_context": list(context)} if context else chunk


def _opening(completion: Completion, show_precontext: bool) -> list[dict]:
    """Role chunk, then the precontext block if the caller asked to see it.

    Precontext goes before any content so a client can parse and strip it without
    buffering the whole response. It is omitted entirely when unrequested: it travels
    inside `content`, so emitting it regardless splices tool JSON into the prose of
    every caller who never asked for it.
    """
    from .envelope import precontext_delta

    chunks = [_role_chunk(completion)]
    # Never an empty one: each relay round opened with `<precontext> [] </precontext>`,
    # and the relay's log of the answer began with two of them before the JSON.
    if show_precontext and completion.precontext:
        chunks.append(precontext_delta(completion))
    return chunks


def _content_chunk(completion: Completion, text: str) -> str:
    return sse({"id": completion.request_id, "object": "chat.completion.chunk",
                "created": completion.created, "model": completion.model,
                "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]})


def _add_usage(completion: Completion, counts: dict) -> None:
    completion.usage.prompt_tokens += counts.get("prompt_tokens", 0)
    completion.usage.completion_tokens += counts.get("completion_tokens", 0)
    completion.usage.reasoning_tokens += (
        counts.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)


async def _stream_turns(brain: BrainClient, convo: list[dict], ctx, completion: Completion,
                        caller_tools, sampling: Sampling, tool_choice, refs,
                        show_precontext: bool) -> AsyncIterator[str]:
    """A streamed answer: the tool loop runs and its text is forwarded as generated.

    The loop runs as a task and hands text over a queue, so chunks leave while the
    model is still writing. Precontext is sent before the first text that follows a
    tool, as the buffered stream always sent it before content.
    """
    from .envelope import precontext_delta, tool_call_deltas

    queue: asyncio.Queue = asyncio.Queue()

    async def run() -> None:
        try:
            result = await _run_tool_loop(brain, convo, ctx, completion.usage, caller_tools,
                                          sampling, tool_choice=tool_choice,
                                          on_text=queue.put_nowait,
                                          precontext=completion.precontext)
            queue.put_nowait(("done", result))
        except _CallerToolCalls as handoff:
            queue.put_nowait(("calls", handoff))
        except Exception as exc:  # reported in-band below; the status line is sent
            queue.put_nowait(("error", exc))

    task = asyncio.create_task(run())
    shown = 0
    spoke = False

    def unshown_precontext() -> list[str]:
        nonlocal shown
        if not show_precontext or len(completion.precontext) <= shown:
            return []
        fresh = Completion(request_id=completion.request_id, model=completion.model,
                           created=completion.created, precontext=completion.precontext[shown:])
        shown = len(completion.precontext)
        return [sse(precontext_delta(fresh))]

    try:
        yield sse(_role_chunk(completion))
        while True:
            item = await queue.get()
            if isinstance(item, str):
                spoke = True
                for chunk in unshown_precontext():
                    yield chunk
                yield _content_chunk(completion, item)
                continue
            kind, value = item
            if kind == "done":
                for chunk in unshown_precontext():
                    yield chunk
                finish = value[4]
                if not spoke:
                    # The last turn ended with neither text nor a tool call. Answer from
                    # what the tools returned, as the buffered path does, rather than
                    # close the stream empty.
                    outcome = {"finish_reason": "stop"}
                    async for piece in brain.synthesise_stream(
                            value[0], sampling=sampling,
                            on_finish=lambda reason: outcome.update(finish_reason=reason),
                            on_usage=lambda counts: _add_usage(completion, counts)):
                        yield _content_chunk(completion, piece)
                    finish = outcome["finish_reason"]
                yield sse(_final_chunk(completion, finish))
            elif kind == "calls":
                for chunk in unshown_precontext():
                    yield chunk
                for chunk in tool_call_deltas(completion, value.calls):
                    yield sse(chunk)
                yield sse(_with_context(_final_chunk(completion, "tool_calls"), value.context))
            else:
                log.error("streaming failed mid-response", exc_info=value)
                yield _stream_failure(completion)
            yield sse("[DONE]")
            return
    finally:
        if not task.done():
            task.cancel()
        refs.cleanup()


async def _stream_tool_calls(completion: Completion, calls, refs,
                             show_precontext: bool, context=()) -> AsyncIterator[str]:
    """A tool-call turn as SSE. Content stays empty; the call rides in its own delta."""
    from .envelope import tool_call_deltas

    try:
        for chunk in _opening(completion, show_precontext):
            yield sse(chunk)
        for chunk in tool_call_deltas(completion, calls):
            yield sse(chunk)
        yield sse(_with_context(_final_chunk(completion, "tool_calls"), context))
        yield sse("[DONE]")
    finally:
        refs.cleanup()


async def _stream_text(completion: Completion, text: str, refs,
                       show_precontext: bool,
                       finish_reason: str = "stop") -> AsyncIterator[str]:
    """Deliver already-generated text as SSE.

    Used where generation cannot be incremental -- guided decoding has to constrain the
    whole object -- but the caller still asked for a stream. The transport the caller
    chose is honoured even when the generation behind it was not incremental.
    """
    try:
        for chunk in _opening(completion, show_precontext):
            yield sse(chunk)
        if text:
            yield _content_chunk(completion, text)
        yield sse(_final_chunk(completion, finish_reason))
        yield sse("[DONE]")
    finally:
        refs.cleanup()


async def _stream_response(
    brain: BrainClient, convo: list[dict], completion: Completion, refs,
    think: bool = False, effort: str | None = None,
    sampling: Sampling | None = None, show_precontext: bool = False,
) -> AsyncIterator[str]:
    """SSE: role chunk, precontext chunk, content deltas, finish, [DONE]."""
    try:
        for chunk in _opening(completion, show_precontext):
            yield sse(chunk)

        # Filled in by the generation itself. Defaulting to "stop" and never revising
        # it is what made a truncated answer indistinguishable from a complete one.
        outcome = {"finish_reason": "stop"}

        def note_finish(reason: str) -> None:
            outcome["finish_reason"] = reason

        async for piece in brain.synthesise_stream(
                convo, think=think, effort=effort, sampling=sampling,
                on_finish=note_finish, on_usage=lambda counts: _add_usage(completion, counts)):
            yield _content_chunk(completion, piece)

        yield sse(_final_chunk(completion, outcome["finish_reason"]))
        yield sse("[DONE]")
    except Exception:
        log.exception("streaming failed mid-response")
        yield _stream_failure(completion)
        yield sse("[DONE]")
    finally:
        refs.cleanup()


def _parses(text: str) -> bool:
    try:
        json.loads(text)
    except (TypeError, ValueError):
        return False
    return True


# What ocr adds to its result when asked for bounds: every block and line on the page with
# its coordinates.
_PAGE_GEOMETRY = ("layout_note", "layout", "lines", "layout_truncated", "lines_truncated",
                  "lines_omitted")


def _without_page_geometry(messages: list[dict]) -> list[dict]:
    """Tool results without the page geometry ocr adds for bounds, to fill a schema that asks
    for no coordinates. When the model had asked for bounds, hundreds of boxed blocks and
    lines sat beside the text, and a paper's section list was filled empty."""
    out = []
    for message in messages:
        content = message.get("content")
        if message.get("role") == "tool" and isinstance(content, str) and '"layout_note"' in content:
            try:
                data = json.loads(content)
            except ValueError:
                out.append(message)
                continue
            if isinstance(data, dict):
                kept = {k: v for k, v in data.items() if k not in _PAGE_GEOMETRY}
                message = {**message, "content": json.dumps(kept, ensure_ascii=False)}
        out.append(message)
    return out


_GEOMETRY_KEYS = ("bbox", "bounds", "top_left", "bottom_right", "top_left_x", "x1",
                  "coordinates", "polygon", "rect", "position")


def _schema_wants_geometry(schema: dict | None) -> bool:
    """Does the caller's response schema ask for coordinates anywhere in it?

    A schema with `top_left_x` in it cannot be filled without boxes, so whether the
    model remembered to request them is not really a judgement call. Walking the
    schema settles it before the tool runs.
    """
    if not schema:
        return False

    stack = [schema]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "properties" and isinstance(value, dict):
                    for name in value:
                        lowered = name.lower()
                        if any(g in lowered for g in _GEOMETRY_KEYS):
                            return True
                stack.append(value)
        elif isinstance(node, list):
            stack.extend(node)
    return False


def _error(message: str, kind: str, request_id: str) -> dict[str, Any]:
    return {"error": {"message": message, "type": kind, "code": None, "request_id": request_id}}


def _stream_failure(completion: Completion) -> str:
    """A failure after the status line was sent, so it can only be reported in-band.

    In the OpenAI error shape, which OpenAI SDKs raise on and the interfaze relay logs as
    a failure. Sent as answer text, it was billed as a success and showed the caller the
    raw exception.
    """
    return sse(_error("The response failed while streaming. Please try again.",
                      "server_error", completion.request_id))
