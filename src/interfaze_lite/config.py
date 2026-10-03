"""Runtime settings, all environment-driven so the container needs no config file."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass
class Settings:
    # vLLM's OpenAI-compatible server. Bound to loopback in the container because its
    # /sleep and /collective_rpc dev routes are unauthenticated.
    brain_url: str = field(default_factory=lambda: _env("BRAIN_URL", "http://127.0.0.1:8001"))
    # The name vLLM ANSWERS to. `vllm serve --served-model-name X` makes the repo id a
    # 404: requests must carry X. Keeping these separate means the served name can stay
    # stable while the checkpoint underneath changes.
    brain_served_name: str = field(
        default_factory=lambda: _env("BRAIN_SERVED_NAME", "interfaze-lite"))
    perception_url: str = field(
        default_factory=lambda: _env("PERCEPTION_URL", "http://127.0.0.1:8002"))
    diarize_url: str = field(
        default_factory=lambda: _env("DIARIZE_URL", "http://127.0.0.1:8003"))

    model_name: str = field(default_factory=lambda: _env("MODEL_NAME", "interfaze-lite"))

    # A tool-calling turn that never converges would otherwise loop until the request
    # times out; interfaze caps steps the same way.
    max_tool_steps: int = field(default_factory=lambda: _int("MAX_TOOL_STEPS", 8))
    max_new_tokens: int = field(default_factory=lambda: _int("MAX_NEW_TOKENS", 4096))
    # Structured extraction needs far more room than prose. Word-level bounding boxes
    # for a single receipt run past 8k tokens, and guided decoding that hits the cap
    # emits a syntactically incomplete object -- the caller gets a JSON parse error
    # rather than a short answer, so the usual "truncation is survivable" logic does
    # not apply here. 16384 cut off a ten-page "every element with its box" request at
    # 20.7k tokens where interfaze wrote 34k, so this is the published ceiling. It fits:
    # tool output is trimmed to max_context_chars, which keeps the prompt near 70k of
    # the brain's 131k context.
    max_structured_tokens: int = field(
        default_factory=lambda: _int("MAX_STRUCTURED_TOKENS", 32000))
    # The largest completion a caller may ask for. This is the advertised ceiling, not
    # the default budget above: those two answer different questions, and validating a
    # request against the internal default rejected the 32000 the SDKs send by default.
    # Matches the max_completion_tokens interfaze publishes for the same endpoint.
    # Outlines require the segmentation pass that boxes alone do not, so this trades
    # latency for shape. Off restores box-only detection.
    detection_outlines: bool = field(
        default_factory=lambda: os.environ.get("DETECTION_OUTLINES", "1") != "0")
    max_output_tokens: int = field(
        default_factory=lambda: _int("MAX_OUTPUT_TOKENS", 32000))
    # Reuse the tool-selection turn's answer instead of generating a second one. Off
    # restores the older behaviour, which regenerated every tool-using answer; kept as
    # a switch because the two turns are produced under different conditions and a
    # quality regression would be worth backing out without a deploy of new code.
    reuse_tool_turn_answer: bool = field(
        default_factory=lambda: os.environ.get("REUSE_TOOL_TURN_ANSWER", "1") != "0")
    request_timeout_s: int = field(default_factory=lambda: _int("REQUEST_TIMEOUT_S", 900))
    tool_timeout_s: int = field(default_factory=lambda: _int("TOOL_TIMEOUT_S", 600))


    # Geometry the model sees. A dense 50-page scan yields thousands of lines; past a
    # few hundred the boxes crowd out the text they describe.
    max_layout_blocks: int = field(
        default_factory=lambda: _int("MAX_LAYOUT_BLOCKS", 400))

    # A ceiling on what the vision tower sees, so prefill cannot grow without bound
    # on a large screenshot. It is NOT a quality fix: an earlier note here claimed
    # high resolution caused runaway box counts, but the logs showed the offending
    # images had already been resized to ~1.5 MP before the model saw them and still
    # differed wildly -- they were upscaled fakes, and the model was reacting to
    # interpolation blur. Set high enough to leave real screenshots untouched;
    # ScreenSpot-v2 tops out near 2 MP.
    ground_max_pixels: int = field(
        default_factory=lambda: _int("GROUND_MAX_PIXELS", 4_194_304))
    # Photographs, separately. On a 1.48 MP photo asking for the sword, the native image
    # put the box on the armour's chest and "every object" looped to the token cap;
    # at 1 MP the sword was found where it is and "every object" returned 12 clean
    # boxes. Large photos looped far less at 1 MP too. The UI budget above is tuned
    # for ScreenSpot and stays.
    object_ground_max_pixels: int = field(
        default_factory=lambda: _int("OBJECT_GROUND_MAX_PIXELS", 1_048_576))

    # The caller's images, shown to the brain beside a tool's reading of them. What the
    # text cannot carry -- a highlighted line, a colour, a crossed-out item -- is only
    # in the picture. 0 withholds them.
    answer_image_max_pixels: int = field(
        default_factory=lambda: _int("ANSWER_IMAGE_MAX_PIXELS", 1_048_576))

    max_context_chars: int = field(
        default_factory=lambda: _int("MAX_CONTEXT_CHARS", 120_000))

    # Read from the environment at boot. Supplied by the deployment rather than
    # written here, so this file states no dependency on any particular provider.
    component_names: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            n for n in os.environ.get("COMPONENT_NAMES", "").split(",") if n.strip()))

    hf_token: str | None = field(default_factory=lambda: os.environ.get("HF_TOKEN"))

    # Unset means open. interfaze-lite is a container you run yourself, and demanding
    # a key from your own localhost is friction with no security value. Set it on any
    # deployment that is actually reachable from outside.
    api_key: str | None = field(default_factory=lambda: os.environ.get("API_KEY") or None)
    # The hosted deployment's key, sent by the interfaze API in `x-api-admin-key` as it is to
    # every model service. Some secret stores expose it in lowercase.
    admin_key: str | None = field(
        default_factory=lambda: os.environ.get("ADMIN_KEY") or os.environ.get("admin_key") or None)

    @property
    def brain_chat_url(self) -> str:
        return f"{self.brain_url.rstrip('/')}/v1/chat/completions"


settings = Settings()
