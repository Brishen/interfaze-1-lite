"""The brain behind llama-server, and the guard answered by the brain.

llama-server speaks the same OpenAI API as vLLM, with three differences the brain client
absorbs: it reads tool_choice as a string only, it fetches image URLs itself, and it words
a too-long prompt differently. The text guard has no model of its own; the brain answers
in Llama Guard's format and guard.verdict reads that unchanged.
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import io
import json

import httpx
import pytest

from interfaze_lite import guard
from interfaze_lite.brain import BrainClient, ContextLengthError
from interfaze_lite.config import Settings


def _png() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (4, 3), "red").save(buf, format="PNG")
    return buf.getvalue()


def _reply(content: str = "ok") -> dict:
    return {"choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 1}}


def _client(handler, backend: str = "llamacpp") -> BrainClient:
    settings = dataclasses.replace(Settings(), brain_backend=backend)
    return BrainClient(httpx.AsyncClient(transport=httpx.MockTransport(handler)), settings)


def _run(coro):
    return asyncio.run(coro)


TOOLS = [{"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}
         for name in ("ocr", "translate")]


class TestToolChoice:
    def test_a_named_function_becomes_required_with_only_that_tool(self):
        sent = []
        brain = _client(lambda r: sent.append(json.loads(r.content)) or httpx.Response(200, json=_reply()))
        _run(brain.select_tools([{"role": "user", "content": "hi"}], TOOLS,
                                tool_choice={"type": "function", "function": {"name": "translate"}}))
        assert sent[0]["tool_choice"] == "required"
        assert [t["function"]["name"] for t in sent[0]["tools"]] == ["translate"]

    def test_vllm_gets_the_named_choice_as_is(self):
        sent = []
        brain = _client(lambda r: sent.append(json.loads(r.content)) or httpx.Response(200, json=_reply()),
                        backend="vllm")
        choice = {"type": "function", "function": {"name": "translate"}}
        _run(brain.select_tools([{"role": "user", "content": "hi"}], TOOLS, tool_choice=choice))
        assert sent[0]["tool_choice"] == choice and len(sent[0]["tools"]) == 2


class TestRemoteImages:
    def test_an_image_url_is_fetched_once_and_sent_inline(self):
        png, sent, fetched = _png(), [], []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET":
                fetched.append(str(request.url))
                return httpx.Response(200, content=png)
            sent.append(json.loads(request.content))
            return httpx.Response(200, json=_reply())

        brain = _client(handler)
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "what is this?"},
            {"type": "image_url", "image_url": {"url": "https://cdn.example.com/a.png"}}]}]
        _run(brain.synthesise(messages))
        _run(brain.synthesise(messages))

        url = sent[0]["messages"][0]["content"][1]["image_url"]["url"]
        assert url == "data:image/png;base64," + base64.b64encode(png).decode()
        assert fetched == ["https://cdn.example.com/a.png"]
        # The caller's conversation is left as it was.
        assert messages[0]["content"][1]["image_url"]["url"] == "https://cdn.example.com/a.png"

    def test_vllm_fetches_its_own(self):
        sent = []
        brain = _client(lambda r: sent.append(json.loads(r.content)) or httpx.Response(200, json=_reply()),
                        backend="vllm")
        _run(brain.synthesise([{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://cdn.example.com/a.png"}}]}]))
        assert sent[0]["messages"][0]["content"][0]["image_url"]["url"] == "https://cdn.example.com/a.png"


@pytest.mark.parametrize("stream", [False, True])
def test_llama_servers_context_error_is_the_callers(stream):
    body = {"error": {"code": 400, "type": "exceed_context_size_error", "message": (
        "request (70052 tokens) exceeds the available context size (65536 tokens), try increasing it")}}
    brain = _client(lambda _: httpx.Response(400, json=body))
    messages = [{"role": "user", "content": "long"}]

    async def go():
        if stream:
            async for _ in brain.synthesise_stream(messages):
                pass
        else:
            await brain.synthesise(messages)

    with pytest.raises(ContextLengthError, match="exceeds the available context size"):
        _run(go())


class TestTextGuard:
    def test_the_prompt_carries_the_taxonomy_and_the_message(self):
        text = guard.check_prompt("How do I bake bread?")
        assert "S1: Violent Crimes." in text and "S14: Code Interpreter Abuse." in text
        assert "<BEGIN USER MESSAGE>\nHow do I bake bread?\n<END USER MESSAGE>" in text

    def test_only_the_end_of_a_long_message_is_classified(self):
        text = guard.check_prompt("a" * guard.MAX_CHARS + "TAIL")
        assert "TAIL" in text and "a" * guard.MAX_CHARS not in text

    @pytest.mark.parametrize(("answer", "expected"), [
        ("safe", "safe"),
        ("unsafe\nS9", "unsafe\nS9"),
        ("unsafe\nS1, S10", "unsafe\nS1,S10"),
        ("**unsafe**\nS2,S2", "unsafe\nS2"),
        ("unsafe", "unsafe"),
        ("<think>hmm</think>\nsafe", "safe"),
        ("", "safe"),
        ("I cannot help with that.", "safe"),
        ("unsafe\nS15, S0, S14", "unsafe\nS14"),
    ])
    def test_the_answer_is_read_in_llama_guards_format(self, answer, expected):
        assert guard.parse_check(answer) == expected

    def test_the_brain_answers_and_verdict_reads_it(self):
        sent = []
        brain = _client(lambda r: sent.append(json.loads(r.content))
                        or httpx.Response(200, json=_reply("unsafe\nS1")))
        result = _run(brain.text_safety("How to kill a human?"))
        assert result == {"output": "unsafe\nS1", "prompt_tokens": 7, "completion_tokens": 1}
        assert sent[0]["chat_template_kwargs"] == {"enable_thinking": False}
        assert sent[0]["max_tokens"] == 24 and sent[0]["temperature"] == 0
        assert guard.verdict(result["output"], ["S1", "S10"])[:2] == (False, "unsafe S1")

