"""Wire-format tests: the OpenAI envelope plus interfaze's precontext extension."""

import json

from interfaze_lite.envelope import (
    Completion,
    PrecontextItem,
    ToolCall,
    Usage,
    parse_precontext,
    precontext_delta,
    stream,
    strip_precontext,
)


def make(**kw) -> Completion:
    return Completion(request_id="req-test", created=1700000000, **kw)


class TestNonStreaming:
    def test_basic_shape_matches_openai(self):
        body = make(usage=Usage(10, 5)).message("hello")
        assert body["object"] == "chat.completion"
        assert body["model"] == "interfaze-lite"
        assert body["usage"] == {
            "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
            # Reported even at zero: clients read reasoning tokens from here, and an
            # absent key looks identical to a broken mapping on their side.
            "completion_tokens_details": {"reasoning_tokens": 0, "cached_tokens": 0},
            "prompt_tokens_details": {"cached_tokens": 0},
        }
        choice = body["choices"][0]
        assert choice["message"] == {"role": "assistant", "content": "hello"}
        assert choice["logprobs"] is None
        assert choice["finish_reason"] == "stop"

    def test_tool_call_turn_has_null_content(self):
        """Populating content on a tool-call turn breaks OpenAI-compatible agent loops."""
        body = make().tool_calls([ToolCall("call_1", "ocr", '{"file_ref_id":"ref-0"}')])
        msg = body["choices"][0]["message"]
        assert msg["content"] is None
        assert body["choices"][0]["finish_reason"] == "tool_calls"
        assert msg["tool_calls"][0]["type"] == "function"
        assert msg["tool_calls"][0]["function"]["name"] == "ocr"
        # arguments must be a JSON *string*, not an object
        assert isinstance(msg["tool_calls"][0]["function"]["arguments"], str)

    def test_precontext_carries_structured_results(self):
        body = make(precontext=[PrecontextItem("ocr", {"sections": [{"text": "hi"}]})]).message("hi")
        assert body["precontext"] == [{"name": "ocr", "result": {"sections": [{"text": "hi"}]}}]

    def test_precontext_key_always_present_even_when_empty(self):
        """A stable key beats a conditional one: callers never need to branch, and an
        SDK modelling the response as a typed object reports [] rather than missing."""
        body = make().message("hello")
        assert body["precontext"] == []

    def test_interfaze_cache_fields_are_always_present(self):
        """Interfaze always sends vcache and the cached-token details; so does lite, as
        false and 0, since a caller written against interfaze reads them."""
        body = make(precontext=[PrecontextItem("ocr", {})]).message("x")
        assert body["vcache"] is False
        assert body["usage"]["prompt_tokens_details"] == {"cached_tokens": 0}
        assert body["usage"]["completion_tokens_details"]["cached_tokens"] == 0

    def test_optional_fields_appear_only_when_set(self):
        plain = make().message("x")
        assert "reasoning" not in plain and "debug" not in plain
        rich = make(reasoning="thought", debug={"preConfig": {}}).message("x")
        assert rich["reasoning"] == "thought"
        assert rich["debug"] == {"preConfig": {}}


class TestStreaming:
    def test_sequence_order_and_termination(self):
        events = list(stream(make(usage=Usage(3, 4)), ["Hel", "lo"]))
        assert events[-1] == "data: [DONE]\n\n"
        payloads = [json.loads(e[len("data: "):]) for e in events[:-1]]
        assert payloads[0]["choices"][0]["delta"]["role"] == "assistant"
        # Content deltas come after the always-present precontext block, so assert the
        # sequence rather than fixed indices -- otherwise adding a chunk breaks the test
        # without anything actually being wrong.
        contents = [p["choices"][0]["delta"].get("content") or "" for p in payloads]
        assert [c for c in contents if c and "<precontext>" not in c] == ["Hel", "lo"]
        final = payloads[-1]
        assert final["choices"][0]["finish_reason"] == "stop"
        assert final["usage"]["total_tokens"] == 7
        assert all(p["object"] == "chat.completion.chunk" for p in payloads)

    def test_precontext_rides_as_a_sentinel_tagged_content_delta(self):
        """SSE chunks cannot carry top-level fields, so interfaze smuggles precontext
        through the content stream. Byte-compatibility here is part of the contract."""
        c = make(precontext=[PrecontextItem("ocr", {"sections": []})])
        events = list(stream(c, ["done"]))
        text = "".join(
            json.loads(e[len("data: "):])["choices"][0]["delta"].get("content") or ""
            for e in events[:-1]
        )
        assert text.startswith("<precontext> ")
        assert " </precontext>" in text
        assert parse_precontext(text) == [{"name": "ocr", "result": {"sections": []}}]
        assert strip_precontext(text) == "done"

    def test_precontext_block_present_even_with_no_tools(self):
        events = list(stream(make(), ["hi"]))
        text = "".join(
            json.loads(e[len("data: "):])["choices"][0]["delta"].get("content") or ""
            for e in events[:-1]
        )
        assert parse_precontext(text) == []
        assert strip_precontext(text) == "hi"

    def test_precontext_delta_is_emitted_when_empty(self):
        chunk = precontext_delta(make())
        assert "<precontext> [] </precontext>" in chunk["choices"][0]["delta"]["content"]

    def test_empty_content_parts_are_skipped(self):
        events = list(stream(make(), ["", "a", ""]))
        contents = [
            json.loads(e[len("data: "):])["choices"][0]["delta"].get("content") or ""
            for e in events[:-1]
        ]
        real = [c for c in contents if c and "<precontext>" not in c]
        # the two empty parts are dropped; only "a" survives
        assert real == ["a"]


class TestPrecontextParsing:
    def test_returns_none_without_sentinels(self):
        assert parse_precontext("just prose") is None

    def test_returns_none_on_unterminated_block(self):
        assert parse_precontext("<precontext> {broken") is None

    def test_strip_is_a_noop_without_sentinels(self):
        assert strip_precontext("just prose") == "just prose"
