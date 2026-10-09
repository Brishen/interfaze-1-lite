"""Client for the vLLM-served brain, and the multi-stage loop around it.

Interfaze runs four brain stages: route, call tools, synthesise, shape to schema.
Lite collapses the first two. Routing in interfaze picks *which provider* to use --
gpt-5 versus gemini-flash versus glm -- and that decision does not exist here, because
there is one model. What remains of stage one is tool selection, which the tool-calling
call already does.

Three stages, and each uses a different vLLM mode for a specific reason:

  select_and_run  streamed for a streamed request, tools passed, the caller's tool_choice
  synthesise      streaming, NO tools passed
  structured      guided_json against the caller's schema

Tool turns were once never streamed: vLLM leaked raw <tool_call> XML into streamed
content when tools were passed. On 0.27.1 that is gone -- measured directly, a streamed
turn gives clean content deltas and intact tool calls -- so a streamed request streams
its tool turns too, and the answer arrives token by token.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field, replace
from typing import Any

import httpx

from . import grounding, guard
from .config import Settings
from .contracts import InputFetchError
from .envelope import ToolCall

log = logging.getLogger("interfaze.brain")


class ContextLengthError(ValueError):
    """The request is longer than the model's context window."""


class BrainError(RuntimeError):
    pass


class ToolCallCutOff(BrainError):
    """The model's tool call ran past the output token limit before it was complete.

    vLLM reports this as finish_reason "length" with the partial call. llama-server
    instead fails the whole response -- a 500, or an error event mid-stream -- because
    it cannot parse the unfinished arguments, so it is recognised here and raised as
    what it is, for the tool loop to recover from.
    """


_CUT_OFF_CALL = "Failed to parse tool call arguments"


def _failure(status: int, text: str) -> Exception:
    """The error a failed brain response stands for."""
    if (too_long := _context_error(status, text)) is not None:
        return too_long
    if _CUT_OFF_CALL in text:
        return ToolCallCutOff(text[:800])
    return BrainError(f"brain returned {status}: {text[:800]}")


# How each server says the prompt does not fit: vLLM, then llama-server (message, then
# error type).
_CONTEXT_ERRORS = ("maximum context length", "exceeds the available context size",
                   "exceed_context_size_error")


def _context_error(status: int, text: str) -> ContextLengthError | None:
    if status != 400 or not any(marker in text for marker in _CONTEXT_ERRORS):
        return None
    try:
        message = json.loads(text).get("error", {}).get("message") or text
    except (ValueError, AttributeError):
        message = text
    return ContextLengthError(message[:800])


# This model's chat template accepts only these three levels and raises outright on
# anything else -- including "high", which is the value the OpenAI vocabulary (and
# therefore every SDK) actually sends. Forwarding the caller's string unmapped turns a
# routine request into a 400 from the template engine.
# The template accepts only these three. `high` is not one of them, and mapping it to
# the top of the range is the obvious reading but the wrong one: the checkpoint's own
# tracker reports xhigh returning an empty answer with finish_reason "stop" on roughly
# one call in six, and the upstream guidance is to resolve `high` to `medium`. An
# effort level that sometimes answers with nothing is worse than a shallower one.
_EFFORT = {"low": "low", "medium": "medium", "high": "medium", "xhigh": "xhigh"}


def thinking_kwargs(think: bool, effort: str | None = None) -> dict[str, Any]:
    """Chat-template switches for a thinking turn.

    Thinking is requested here but this checkpoint does not currently produce a
    separable reasoning block. Measured directly, with the reasoning parser removed so
    the generation arrives unparsed: the output carries neither `<think>` nor
    `</think>`. The template opens the block in the prompt, so the model is generating
    inside it and simply never closes it, and vLLM's parser reads a missing end tag as
    "thinking is off" and routes everything to content.

    That is upstream behaviour, not something this service can correct -- it is also
    why `reasoning` comes back empty and why reasoning_tokens is zero. Ruled out along
    the way: the tool-call parser consuming the block (identical output under
    qwen3_coder and qwen3_xml), the effort level (all three behave the same), and
    greedy decoding (unchanged at temperature 0.6).
    """
    if not think:
        return {"enable_thinking": False}
    return {"enable_thinking": True,
            # Unrecognised effort falls back to the same level `high` resolves to,
            # for the same reason: the template's own default is xhigh, and xhigh is
            # the level that intermittently answers with nothing.
            "reasoning_effort": _EFFORT.get((effort or "").lower(), "medium")}


# The most randomness a structured request is given, whatever the caller asked for: in its
# tool turns and in filling the schema.
STRUCTURED_MAX_TEMPERATURE = 0.3


@dataclass
class Sampling:
    """What the caller asked for, as opposed to what this service defaults to.

    `max_tokens` was previously fixed at the service default on every path, so a
    request that set it was generated at full length and always finished with "stop" --
    the caller could neither cap cost nor detect truncation. None means "use the
    service default", which is not the same as 0.
    """

    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None

    def apply(self, payload: dict[str, Any], default_max: int) -> dict[str, Any]:
        payload["max_tokens"] = self.max_tokens or default_max
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        return payload


@dataclass
class BrainReply:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"
    reasoning: str = ""
    reasoning_tokens: int = 0


class BrainClient:
    def __init__(self, http: httpx.AsyncClient, settings: Settings):
        self.http = http
        self.settings = settings
        # Remote images already inlined for llama-server, by URL. A tool loop sends the
        # same conversation several times; this keeps it to one download per image.
        self._inlined: dict[str, str] = {}

    async def _adapt(self, payload: dict[str, Any]) -> dict[str, Any]:
        """The request as the configured server can take it. vLLM takes it as it is."""
        if not self.settings.llamacpp:
            return payload
        payload = dict(payload)
        # llama-server reads tool_choice as a string only; a named function is silently
        # read as "auto", so the forced first turn was not forced. Offering only that
        # function and requiring a call is the same constraint.
        choice = payload.get("tool_choice")
        if isinstance(choice, dict):
            name = (choice.get("function") or {}).get("name")
            named = [t for t in payload.get("tools") or []
                     if (t.get("function") or {}).get("name") == name]
            if named:
                payload["tools"] = named
            payload["tool_choice"] = "required"
        if "messages" in payload:
            payload["messages"] = [await self._inline_message(m) for m in payload["messages"]]
        return payload

    async def _inline_message(self, message: dict[str, Any]) -> dict[str, Any]:
        """The message with every remote image as a data URI.

        llama-server downloads an http(s) image_url itself, without the browser
        User-Agent several CDNs insist on, and a failed download fails the whole turn.
        Fetched here instead, as _image_bytes fetches everything else.
        """
        content = message.get("content")
        if not isinstance(content, list) or not any(
                isinstance(p, dict) and p.get("type") == "image_url"
                and str((p.get("image_url") or {}).get("url", "")).startswith(("http://", "https://"))
                for p in content):
            return message
        parts = []
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "image_url":
                parts.append(part)
                continue
            url = str((part.get("image_url") or {}).get("url", ""))
            if url.startswith(("http://", "https://")):
                if url not in self._inlined:
                    if len(self._inlined) >= 32:
                        self._inlined.pop(next(iter(self._inlined)))
                    raw = await self._image_bytes(url)
                    self._inlined[url] = await asyncio.to_thread(_data_uri, raw)
                part = {**part, "image_url": {**part["image_url"], "url": self._inlined[url]}}
            parts.append(part)
        return {**message, "content": parts}

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload = await self._adapt(payload)
        resp = await self.http.post(
            self.settings.brain_chat_url,
            json={"model": self.settings.brain_served_name, **payload},
            timeout=self.settings.request_timeout_s,
        )
        # The caller's input is too long for the model: their error, which a 502 told
        # them to retry.
        if resp.status_code >= 400:
            raise _failure(resp.status_code, resp.text)
        return resp.json()

    async def _events(self, payload: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """A streamed completion, one parsed server-sent event at a time.

        Consume it under `contextlib.aclosing`: closing it closes the connection, which
        is how a caller that stops early makes vLLM abort the generation.
        """
        payload = await self._adapt(payload)
        async with self.http.stream(
            "POST", self.settings.brain_chat_url,
            json={"model": self.settings.brain_served_name, **payload, "stream": True},
            timeout=self.settings.request_timeout_s,
        ) as resp:
            if resp.status_code >= 400:
                body = (await resp.aread()).decode(errors="replace")
                raise _failure(resp.status_code, body)
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                # A failure after the status line arrives as an error event.
                if isinstance(event, dict) and event.get("error"):
                    raise _failure(500, data)
                yield event

    @staticmethod
    def _parse(body: dict[str, Any]) -> BrainReply:
        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = body.get("usage") or {}

        calls = [
            ToolCall(
                id=tc.get("id") or f"call_{i}",
                name=(tc.get("function") or {}).get("name", ""),
                arguments=(tc.get("function") or {}).get("arguments") or "{}",
            )
            for i, tc in enumerate(message.get("tool_calls") or [])
        ]
        # The server runs --reasoning-parser qwen3, which splits thinking out of
        # content into its own field. Surface it separately, and fall back to it when
        # content is empty: returning "" to the caller is worse than returning what the
        # model actually said.
        reasoning = message.get("reasoning_content") or ""
        content = message.get("content") or ""
        if not content.strip() and reasoning.strip():
            content, reasoning = reasoning, ""
        return BrainReply(
            content=content,
            reasoning=reasoning,
            tool_calls=calls,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            finish_reason=choice.get("finish_reason") or "stop",
            reasoning_tokens=(usage.get("completion_tokens_details") or {}).get(
                "reasoning_tokens", 0),
        )

    async def select_tools(self, messages: list[dict], tools: list[dict],
                           sampling: Sampling | None = None, *,
                           tool_choice: str | dict = "auto",
                           on_text: Callable[[str], None] | None = None) -> BrainReply:
        """One turn of the tool loop: it answers, or it asks for tools.

        With `on_text`, the turn streams and each piece of the answer is handed over as
        it is generated, so a streamed request is token by token rather than one block
        at the end. vLLM once leaked raw <tool_call> markup into streamed content when
        tools were passed; on 0.27.1 a streamed turn returns clean deltas and intact
        tool calls, measured directly. The reply is reassembled into the shape the
        buffered path returns and parsed by the same code.
        """
        payload = (sampling or Sampling()).apply({
            "messages": messages,
            "temperature": 0,
            # Thinking is on by default on this model. Deciding which tool to call is
            # not a reasoning task, and the tokens are pure latency here.
            "chat_template_kwargs": {"enable_thinking": False},
            **({"tools": tools, "tool_choice": tool_choice} if tools else {}),
        }, self.settings.max_new_tokens)
        if on_text is None:
            return self._parse(await self._post({**payload, "stream": False}))

        content, reasoning, finish, usage = [], [], None, {}
        calls: dict[int, dict] = {}
        async with contextlib.aclosing(self._events(
                {**payload, "stream_options": {"include_usage": True}})) as events:
            async for chunk in events:
                usage = chunk.get("usage") or usage
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        content.append(delta["content"])
                        on_text(delta["content"])
                    if delta.get("reasoning_content"):
                        reasoning.append(delta["reasoning_content"])
                    for tc in delta.get("tool_calls") or []:
                        slot = calls.setdefault(tc.get("index", 0), {"id": "", "name": "", "args": ""})
                        slot["id"] = tc.get("id") or slot["id"]
                        fn = tc.get("function") or {}
                        slot["name"] += fn.get("name") or ""
                        slot["args"] += fn.get("arguments") or ""
                    finish = choice.get("finish_reason") or finish
        message = {"content": "".join(content), "reasoning_content": "".join(reasoning),
                   "tool_calls": [{"id": c["id"], "function": {"name": c["name"], "arguments": c["args"]}}
                                  for _, c in sorted(calls.items())]}
        return self._parse({"choices": [{"message": message, "finish_reason": finish}], "usage": usage})

    async def synthesise(self, messages: list[dict], *, think: bool = False,
                         effort: str | None = None,
                         sampling: Sampling | None = None) -> BrainReply:
        """Answer. Thinking is opt-in via reasoning_effort, not on by default.

        The model thinks by default, and on a request that needs no deliberation those
        tokens are pure latency -- often more of them than the answer itself.
        """
        sampling = sampling or Sampling()
        payload = sampling.apply({
            "messages": messages,
            "stream": False,
            "temperature": 0,
            "chat_template_kwargs": thinking_kwargs(think, effort),
        }, self.settings.max_new_tokens)
        return self._parse(await self._post(payload))

    async def synthesise_stream(self, messages: list[dict], *, think: bool = False,
                                effort: str | None = None,
                                sampling: Sampling | None = None,
                                on_finish: Callable[[str], None] | None = None,
                                on_usage: Callable[[dict], None] | None = None,
                                ) -> AsyncIterator[str]:
        """Stream the final answer. No tools are passed, so the parser bug cannot fire.

        Thinking, when enabled, arrives on a separate `reasoning_content` delta. It is
        re-wrapped in <think> tags on the way out because that is the side channel the
        interfaze SDKs already know how to strip and expose as `.reasoning` -- a
        client's text_deltas() never sees it.
        """
        sampling = sampling or Sampling()
        payload = sampling.apply({
            "messages": messages,
            "temperature": 0,
            "chat_template_kwargs": thinking_kwargs(think, effort),
            # Ask for the same accounting the buffered path gets. Without it a streamed
            # turn reports zero tokens, so usage depended on which transport was used.
            "stream_options": {"include_usage": True},
        }, self.settings.max_new_tokens)
        thinking = False
        async with contextlib.aclosing(self._events(payload)) as events:
            async for chunk in events:
                choice = (chunk.get("choices") or [{}])[0]
                # The reason the generation ended is only ever stated here. Assuming
                # "stop" meant a caller who set max_tokens could not tell a complete
                # answer from a truncated one.
                if choice.get("finish_reason") and on_finish:
                    on_finish(choice["finish_reason"])
                if chunk.get("usage") and on_usage is not None:
                    on_usage(chunk["usage"])
                delta = (choice.get("delta") or {})
                if delta.get("reasoning_content"):
                    if not thinking:
                        thinking = True
                        yield "<think>"
                    yield delta["reasoning_content"]
                if delta.get("content"):
                    if thinking:
                        thinking = False
                        yield "</think>"
                    yield delta["content"]
        # An unclosed <think> would make the SDK's side-channel filter swallow
        # everything after it.
        if thinking:
            yield "</think>"

    async def structured(self, messages: list[dict], schema: dict,
                         sampling: Sampling | None = None) -> BrainReply:
        """Emit an object conforming to `schema`.

        Uses vLLM's guided decoding, which -- unlike tool_choice="auto" -- genuinely
        constrains generation, so the result is schema-valid by construction rather
        than by hope. Costs under 6% end-to-end latency.
        """
        sampling = sampling or Sampling()
        if sampling.temperature is not None and sampling.temperature > STRUCTURED_MAX_TEMPERATURE:
            # Filling a schema copies what the context says, and sampling only loses or
            # invents things: at the playground's temperature of 1, a paper's section list
            # came back empty on 2 runs of 11 and once listed two sections the page does
            # not have. interfaze-beta gave the same list on every run.
            sampling = replace(sampling, temperature=STRUCTURED_MAX_TEMPERATURE)
        return self._parse(await self._post(sampling.apply({
            "messages": messages,
            "stream": False,
            "temperature": 0,
            "response_format": {"type": "json_schema", "json_schema": schema},
            # No frequency_penalty. It subtracts 0.2 per earlier occurrence, and JSON is
            # made of repeats -- quotes, "},{", the same keys item after item -- so by a
            # few dozen items the tokens that close a string were several logits down,
            # and under batched load the model stayed inside the string: a receipt came
            # back as one item priced "}, {" on 12 of 18 runs with it, 0 of 18 without.
            # A degenerate loop is bounded by max_structured_tokens instead.
            # Thinking is on by default on this model, and the server runs
            # --reasoning-parser qwen3. With both active the parser routes the whole
            # generation into reasoning_content and `content` comes back empty, which
            # is exactly what every JSONDecodeError in the suite was.
            "chat_template_kwargs": {"enable_thinking": False},
        }, self.settings.max_structured_tokens)))

    async def _bounded_image(self, image_url: str, budget: int, *, inline: bool = False,
                             floor: int = 0) -> tuple[str, tuple[int, int] | None]:
        """Resize the image down to the grounding pixel budget before sending it.

        Grounding degrades sharply with resolution, and not gracefully: the same
        three-object photograph answered correctly at 1.5 MP, and at 3.3 MP came back
        with 43 boxes -- deterministically, the same count every run. It is a
        repetition loop rather than slow prefill.

        vLLM's `--mm-processor-kwargs {"max_pixels": ...}` is the natural place for
        this and is silently ignored by some the brain processors; capping it here instead
        means the bound holds whatever the server decides to honour. Images already
        under the budget are returned untouched, so the common path pays nothing.

        Also returns the image's own size, which is all that turning the model's
        0-1000 grid into pixels needs.
        """
        if budget <= 0:
            return image_url, None

        try:
            raw = await self._image_bytes(image_url)
            # Decode and resize are CPU-bound and can run for hundreds of
            # milliseconds on a large page, so they belong off the loop too.
            return await asyncio.to_thread(_shrink_to_budget, raw, budget, image_url, inline, floor)
        except InputFetchError:
            raise
        except Exception as exc:
            # A resize failure must not cost the caller their detection.
            log.warning("could not bound grounding image (%s); sending as-is", exc)
            return image_url, None

    async def _image_bytes(self, image_url: str) -> bytes:
        """The image itself, from a data URI, a URL, or a materialised upload's path."""
        import base64
        import pathlib

        if image_url.startswith("data:"):
            return base64.b64decode(image_url.split(",", 1)[1])
        if not image_url.startswith(("http://", "https://")):
            return await asyncio.to_thread(pathlib.Path(image_url).read_bytes)
        try:
            # The shared httpx client, awaited. This used to be a blocking
            # urllib.urlopen inside an async def, which parks the whole event
            # loop for the length of a download -- with max_inputs=6 that stalls
            # five unrelated requests behind one slow image.
            #
            # The browser User-Agent stays: several CDNs answer "Python-urllib"
            # and friends with 403, and the failure here is silent -- the resize
            # is skipped and the full-resolution image goes to the model anyway.
            response = await self.http.get(image_url, timeout=60, headers={
                "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                               "AppleWebKit/537.36 (KHTML, like Gecko) "
                               "Chrome/125.0 Safari/537.36"),
                "Accept": "image/*,*/*",
            })
        except httpx.RequestError as exc:
            raise InputFetchError(f"could not fetch {image_url}: {type(exc).__name__}") from exc
        if response.status_code >= 400:
            raise InputFetchError(f"could not fetch {image_url}: HTTP {response.status_code}")
        return response.content

    async def viewable(self, image_url: str, budget: int, *, inline: bool = False,
                       floor: int = 0) -> tuple[str, tuple[int, int] | None]:
        """An image as the vision tower should get it: reachable, and within `budget` pixels.

        vLLM fetches image_url on its own, from inside its own process. A local path --
        which is what a data: URI resolves to after filerefs materialises it -- is not
        reachable that way, so it is inlined instead. `inline` inlines every image that
        decodes, so vLLM never fetches -- or fails to -- on its own.
        """
        if not image_url.startswith(("http://", "https://", "data:")):
            # Read and base64-encode off the loop: a materialised upload can be tens
            # of megabytes, and both the read and the encode are blocking.
            image_url = await asyncio.to_thread(_inline_local_file, image_url)
        return await self._bounded_image(image_url, budget, inline=inline, floor=floor)

    async def ground(self, image_url: str, prompts: list[str], *,
                     domain: str = "ui") -> dict[str, Any]:
        """Locate things in an image, using the brain itself.

        Same model that routes and answers -- no separate grounding specialist. That is
        a measured choice: on ScreenSpot-v2 this path scores comparably to a dedicated
        grounding model, without the VRAM a second model would cost.

        `domain` picks the wording, and it matters more than it looks. The UI phrasing
        asks the model to find elements "in this UI screenshot"; pointed at a photograph
        it frequently returns [] because nothing in the image is a UI element. Object
        detection therefore gets neutral wording about an image.
        """
        budget = (self.settings.ground_max_pixels if domain == "ui"
                  else self.settings.object_ground_max_pixels)
        # A small photograph is seen at the budget: at 0.33 MP the crowd demo's cameras
        # were all boxed about 5% too high, and at 1.3 MP they were on target. Boxes are
        # on a relative grid, so they still land on the original.
        viewed, size = await self.viewable(image_url, budget,
                                           floor=budget // 2 if domain == "object" else 0)
        label = ", ".join(prompts)
        question = grounding.prompt(prompts, domain)

        reply, looped = await self._ground_reply(viewed, question, grounding.max_tokens(domain))
        elements = grounding.parse(reply, label, looped=looped)
        if domain == "ui" and size:
            # The original, not the copy shrunk to the budget: the quarters are cut from
            # it to be seen closer, and the margins are trimmed against its pixels.
            try:
                raw = await self._image_bytes(image_url)
            except InputFetchError as exc:
                # Read once already; failing a second time must not cost the detection.
                log.warning("could not read the screenshot again (%s); boxes as grounded", exc)
                raw = None
            if raw is not None and looped:
                # A screenshot crowded enough to loop is grounded again in quarters, which
                # find what the loop crowded out.
                log.info("grounding %r looped on %dx%d; grounding its quarters", label, *size)
                crops = await asyncio.to_thread(_tile_uris, raw, budget)
                replies = await asyncio.gather(*(
                    self._ground_reply(uri, question, grounding.max_tokens(domain)) for uri, _ in crops))
                elements = grounding.suppress_duplicates([
                    e for (_, tile), (text, tile_looped) in zip(crops, replies, strict=True)
                    for e in grounding.from_tile(grounding.parse(text, label, looped=tile_looped), tile, *size)])
            if raw is not None:
                elements = await asyncio.to_thread(_trimmed, elements, raw, budget)
        # The image's own size travels with the boxes, so they become pixels without a
        # second service: through /segment, a busy moment returned grid coordinates as
        # though they were pixels.
        return {"gui_elements": elements,
                **({"width": size[0], "height": size[1]} if size else {})}

    async def _ground_reply(self, image_url: str, question: str, max_tokens: int) -> tuple[str, bool]:
        """The grounding reply, and whether it looped: cut off by the watch or the cap.

        Streamed so a loop is stopped when it starts, not at the token cap.
        """
        payload = {
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": question},
            ]}],
            "max_tokens": max_tokens,
            "temperature": 0,
            # Grounding emits a JSON box list; chain of thought only delays it.
            "chat_template_kwargs": {"enable_thinking": False},
        }
        watch, finish = grounding.LoopWatch(), None
        async with contextlib.aclosing(self._events(payload)) as events:
            async for chunk in events:
                for choice in chunk.get("choices") or []:
                    finish = choice.get("finish_reason") or finish
                    if watch.feed((choice.get("delta") or {}).get("content") or ""):
                        return watch.raw, True
        return watch.raw, finish == "length"

    _IMAGE_SAFETY_SCHEMA = {
        "name": "image_safety",
        "schema": {
            "type": "object",
            "properties": {
                "adult_score": {"type": "number", "description": (
                    "0 to 1: how likely the image is sexually explicit -- nudity (exposed "
                    "genitals, female nipples, buttocks) or sexual activity.")},
                "racy_score": {"type": "number", "description": (
                    "0 to 1: how likely the image is sexually suggestive without being "
                    "explicit -- lingerie, revealing clothing in a sexualised context, "
                    "provocative poses, partial nudity.")},
                "gore_score": {"type": "number", "description": (
                    "0 to 1: how likely the image is gory -- blood, open wounds, graphic "
                    "injury, mutilation, dead bodies.")},
            },
            "required": ["adult_score", "racy_score", "gore_score"],
            "additionalProperties": False,
        },
    }

    async def image_safety(self, image_url: str) -> dict[str, float]:
        """Adult, racy and gore scores for one image, from the brain itself.

        interfaze reads the same three off a hosted adult-content classifier and derives
        nudity, gore and NSFW from them; this deployment has no hosted classifier, and
        its vision model is already on the card. Raises InputFetchError when the image
        cannot be fetched.
        """
        image_url, size = await self.viewable(
            image_url, self.settings.answer_image_max_pixels, inline=True)
        if size is None:
            raise InputFetchError("could not read the image for the safety check")
        reply = await self.structured(
            [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": (
                    "Classify this image for content moderation. Score each category "
                    "independently from 0 (certainly not) to 1 (certainly).")},
            ]}],
            self._IMAGE_SAFETY_SCHEMA, Sampling(max_tokens=200, temperature=0))
        scores = json.loads(reply.content)

        def clamp(value) -> float:
            try:
                return min(1.0, max(0.0, float(value)))
            except (TypeError, ValueError):
                return 0.0

        return {"adult": clamp(scores.get("adult_score")), "racy": clamp(scores.get("racy_score")),
                "gore": clamp(scores.get("gore_score")),
                "prompt_tokens": reply.prompt_tokens, "completion_tokens": reply.completion_tokens}

    async def text_safety(self, text: str) -> dict[str, Any]:
        """The guard's verdict on one user message, from the brain itself.

        `output` is in Llama Guard's format -- "safe", or "unsafe" and the violated codes
        on the next line -- which is what guard.verdict reads. The brain answers the
        same S1-S14 taxonomy a separate guard model was prompted with, so there is no
        guard model to load.
        """
        reply = await self.synthesise(
            [{"role": "user", "content": guard.check_prompt(text)}],
            sampling=Sampling(max_tokens=24, temperature=0))
        return {"output": guard.parse_check(reply.content),
                "prompt_tokens": reply.prompt_tokens, "completion_tokens": reply.completion_tokens}

    async def healthy(self) -> bool:
        try:
            resp = await self.http.get(f"{self.settings.brain_url.rstrip('/')}/health", timeout=5)
            return resp.status_code < 400
        except httpx.HTTPError:
            return False


def _shrink_to_budget(raw: bytes, budget: int, original: str, inline: bool = False,
                      floor: int = 0) -> tuple[str, tuple[int, int]]:
    """A data URI no larger than `budget` pixels (or the original), and the image's size.

    An image under `floor` pixels is scaled up to the budget instead.
    """
    import base64
    import io

    from PIL import Image

    image = Image.open(io.BytesIO(raw))
    pixels = image.width * image.height
    size = (image.width, image.height)
    if 0 < pixels < floor:
        scale = (budget / pixels) ** 0.5
        image = image.convert("RGB").resize(
            (int(image.width * scale), int(image.height * scale)), Image.LANCZOS)
        log.info("grounding: %dx%d scaled up to %dx%d", *size, image.width, image.height)
        return _png_uri(image), size
    if pixels <= budget:
        log.info("grounding: %dx%d within budget", image.width, image.height)
        if inline and not original.startswith("data:"):
            image.load()  # decodes, so a file vLLM cannot read is caught here
            mime = Image.MIME.get(image.format or "", "image/png")
            return f"data:{mime};base64," + base64.b64encode(raw).decode(), size
        return original, size

    image = _within(image.convert("RGB"), budget)
    log.info("grounding: %d px -> %dx%d", pixels, image.width, image.height)
    return _png_uri(image), size


def _tile_uris(raw: bytes, budget: int) -> list[tuple[str, tuple[int, int, int, int]]]:
    """The image's quarters (grounding.tiles), each a data URI within `budget` pixels."""
    import io

    from PIL import Image

    image = Image.open(io.BytesIO(raw)).convert("RGB")
    return [(_png_uri(_within(image.crop(tile), budget)), tile)
            for tile in grounding.tiles(image.width, image.height)]


def _trimmed(elements: list[dict], raw: bytes, budget: int) -> list[dict]:
    """grounding.trim_to_content against the image, seen within `budget` pixels."""
    import io

    from PIL import Image

    try:
        return grounding.trim_to_content(elements, _within(Image.open(io.BytesIO(raw)).convert("RGB"), budget))
    except Exception as exc:
        # Trimming is a refinement; failing at it must not cost the boxes themselves.
        log.warning("could not trim the boxes (%s); returning them as grounded", exc)
        return elements


def _within(image, budget: int):
    """The image scaled down to at most `budget` pixels; itself when it already fits."""
    pixels = image.width * image.height
    if pixels <= budget:
        return image
    scale = (budget / pixels) ** 0.5
    return image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))))


def _png_uri(image) -> str:
    import base64
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _data_uri(raw: bytes) -> str:
    """Image bytes as a data URI, typed by what they decode as."""
    import base64
    import io

    from PIL import Image

    try:
        mime = Image.MIME.get(Image.open(io.BytesIO(raw)).format or "", "image/png")
    except Exception as exc:
        raise InputFetchError(f"not an image ({type(exc).__name__})") from exc
    return f"data:{mime};base64," + base64.b64encode(raw).decode()


def _inline_local_file(path: str) -> str:
    """A local path as a data URI, for a consumer that cannot reach our filesystem."""
    import base64
    import mimetypes
    import pathlib

    blob = pathlib.Path(path).read_bytes()
    mime = mimetypes.guess_type(path)[0] or "image/png"
    return f"data:{mime};base64," + base64.b64encode(blob).decode()
