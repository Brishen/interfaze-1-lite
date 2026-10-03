"""Interfaze Lite -- a composite perception + routing model for `transformers`.

    from transformers import AutoModel
    model = AutoModel.from_pretrained("InterfazeAI/interfaze-lite", trust_remote_code=True)

    model.ocr("invoice.png")                       # sections, lines, words, bounds, confidence
    model.transcribe("call.wav", by_speaker=True)  # speaker-attributed transcript
    model.detect("street.jpg", ["car", "dog"])     # boxes + masks + labels
    model.ground("screen.png", ["the save button"])
    model.chat([{"role": "user", "content": "what does this say?"}], files=["scan.pdf"])

Design notes
------------
Components load lazily. Each accessor materialises its component on first use and
caches it, so an OCR-only caller never pulls the 28 GB brain. Set
`config.lazy_load=False` to force everything up front.

Every method returns the interfaze wire shapes -- four-corner `bounds`, OCR
`sections[].lines[].words[]`, `{text, chunks}` -- via the shared `contracts` module,
so this and the Dockerised service cannot drift apart.

This is the reference implementation. It runs the brain through plain `transformers`,
which means no paged attention and no continuous batching. For throughput, serve the
brain with vLLM using the container in the GitHub repo; the perception components are
identical either way.
"""

from __future__ import annotations

import base64
import concurrent.futures
import gc
import io
import os
import re
import threading
import time
from collections.abc import Sequence
from typing import Any

import torch
from transformers import PreTrainedModel

# trust_remote_code ships a sibling file only when a `from .module import name` line names
# it; `from . import module` is not recognised, and the file never reaches the user. So each
# sibling is imported by name at least once.
try:  # the interfaze_lite package, and this repo as trust_remote_code imports it
    from . import contracts, forecasting, grounding
    from .agent import run as run_agent
    from .configuration_interfaze_lite import InterfazeLiteConfig
    from .contracts import OCRLine
    from .forecasting import MAX_CONTEXT, MAX_HORIZON
    from .grounding import MAX_TOKENS as GROUND_MAX_TOKENS
    from .grounding import MAX_TOKENS_UI as GROUND_MAX_TOKENS_UI
    from .guard import CATEGORIES as GUARD_CATEGORIES
    from .line_runtime import Runtime, layout_pages, line_rows
    from .prompts import NUDGE
except ImportError:  # the files run flat, from a checkout
    import contracts  # type: ignore
    import forecasting  # type: ignore
    import grounding  # type: ignore
    from agent import run as run_agent  # type: ignore
    from configuration_interfaze_lite import InterfazeLiteConfig  # type: ignore
    from contracts import OCRLine  # type: ignore
    from forecasting import MAX_CONTEXT, MAX_HORIZON  # type: ignore
    from grounding import MAX_TOKENS as GROUND_MAX_TOKENS  # type: ignore
    from grounding import MAX_TOKENS_UI as GROUND_MAX_TOKENS_UI  # type: ignore
    from guard import CATEGORIES as GUARD_CATEGORIES  # type: ignore
    from line_runtime import Runtime, layout_pages, line_rows  # type: ignore
    from prompts import NUDGE  # type: ignore


# Some CDNs -- Wikimedia most notably -- reject requests with no User-Agent and return
# an HTML error page with a 200. PIL then fails with the useless "cannot identify image
# file <BytesIO object>", which says nothing about the actual cause.


def _hf_token() -> str | None:
    """The token, under whichever name the environment happens to use.

    Some upstream components are gated. huggingface_hub picks up
    HF_TOKEN automatically, but a Modal secret may expose it under a different name, and
    a silent None turns into "You are trying to access a gated repo" at first inference
    rather than at boot.
    """
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "hf_token"):
        value = os.environ.get(key)
        if value:
            os.environ.setdefault("HF_TOKEN", value)
            return value
    return None


def _fast_asr_enabled() -> bool:
    return os.environ.get("ENABLE_FAST_ASR", "0") not in ("0", "", "false", "False")


# The recogniser's own defences against hallucinated repetition, as in its reference
# decoder: a window whose text compresses too well (a loop) or scores too low is
# re-decoded at a higher temperature. Greedy decoding alone looped "5-5-5" for 60 s
# of a card number; with these the same audio came back digit-for-digit.
_RECOGNISER_GUARDS = {
    "condition_on_prev_tokens": False,
    "compression_ratio_threshold": 1.35,
    "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    "logprob_threshold": -1.0,
    "no_speech_threshold": 0.6,
}


def _language_code(detected: str | None) -> str | None:
    """ISO 639-1, as interfaze reports it. The recogniser names languages ("english")."""
    if not detected or len(detected) <= 3:
        return detected
    from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE

    return TO_LANGUAGE_CODE.get(detected.lower(), detected)


def _segments(words: list) -> list:
    """Sentence-length spans from word timings, for a recogniser that only emits words."""
    spans, current = [], []
    for word in words:
        if current and word.start - current[-1].end > 1.0:
            spans.append(current)
            current = []
        current.append(word)
        if word.text.rstrip().endswith((".", "?", "!")):
            spans.append(current)
            current = []
    if current:
        spans.append(current)
    return [
        contracts.Word(text=" ".join(w.text for w in span), start=span[0].start, end=span[-1].end)
        for span in spans
    ]


def _load_image(src: str | bytes | Any):
    from PIL import Image, UnidentifiedImageError

    if hasattr(src, "convert"):
        return src.convert("RGB")

    if isinstance(src, (bytes, bytearray)):
        data, origin = bytes(src), "<bytes>"
    elif isinstance(src, str) and src.startswith(("http://", "https://")):
        resp = contracts.fetch_url(src, timeout=60, headers={"User-Agent": contracts.UA, "Accept": "image/*"})
        ctype = resp.headers.get("content-type", "")
        if ctype and not ctype.startswith("image/"):
            head = resp.text[:200].replace("\n", " ") if "text" in ctype else ""
            raise ValueError(f"{src} returned content-type {ctype!r}, not an image. {head}".strip())
        data, origin = resp.content, src
    else:
        return Image.open(src).convert("RGB")

    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError(
            f"could not decode an image from {origin} ({len(data)} bytes, starts with {data[:16]!r})"
        ) from exc


class InterfazeLiteModel(PreTrainedModel):
    config_class = InterfazeLiteConfig
    # Components are self-contained checkpoints with their own init; there is no
    # top-level parameter tensor for transformers to initialise.
    _no_split_modules: list[str] = []
    supports_gradient_checkpointing = False

    def __init__(self, config: InterfazeLiteConfig):
        super().__init__(config)
        self.config = config
        self._cache: dict[str, Any] = {}
        # Components are materialised lazily and pages are OCR'd on several threads, so
        # two pages can race to build the same one. Double-loading the line detector or a 9B VLM
        # is expensive and can OOM.
        self._cache_lock = __import__("threading").Lock()
        # The line and layout detectors run in worker processes (line_runtime): as
        # threads they shared Paddle's process-wide oneDNN context, raced on it, and
        # segfaulted the whole perception process under concurrent OCR.
        self._runtime: Runtime | None = None
        self._runtime_lock = threading.Lock()

        self._dtype = (
            getattr(torch, config.torch_dtype) if isinstance(config.torch_dtype, str) else config.torch_dtype
        )
        # This model owns no parameters -- every tensor belongs to a lazily loaded
        # component -- so PreTrainedModel.device raises StopIteration. Resolve the
        # device once, explicitly, and never touch the inherited property.
        self._torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # PyTorch's cuDNN attention backend crashes on some head-dim/sequence shapes with
        #   "Expected mha_graph.execute(...).is_good() to be true, but got false"
        # which is what the document reader's the brain attention hit on this GPU. Turning that one
        # backend off leaves flash and mem-efficient SDPA available, so this costs
        # nothing measurable while removing the crash.
        if torch.cuda.is_available() and hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(False)
        if not config.lazy_load:
            eager = ["brain", "ocr_vlm", "segmenter", "diarizer"]
            if _fast_asr_enabled():
                eager.insert(3, "asr")
            for name in eager:
                self._component(name)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *_model_args,
        config=None,
        token: str | None = None,
        revision: str | None = None,
        **kwargs,
    ):
        """Build the model from its config alone.

        It owns no tensors: every weight belongs to a component in its own folder, loaded
        on first use. The stock loader looks for a weights file at the repo root, finds
        none, and raises.
        """
        if config is None:
            config = cls.config_class.from_pretrained(
                pretrained_model_name_or_path,
                token=token,
                revision=revision,
                trust_remote_code=True,
                **{k: v for k, v in kwargs.items() if k == "cache_dir"},
            )
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        config._name_or_path = str(pretrained_model_name_or_path)
        model = cls(config)
        model.name_or_path = str(pretrained_model_name_or_path)
        model._hub_token = token
        return model

    def _detection_runtime(self) -> Runtime:
        """The pool of detector processes, started on first use."""
        with self._runtime_lock:
            if self._runtime is None:
                self._runtime = Runtime(
                    self.config.line_detector_pool_size,
                    self._line_detector_settings(),
                    self._layout_settings(),
                )
            return self._runtime

    @property
    def device(self):  # type: ignore[override]
        """Override PreTrainedModel.device, which assumes the module owns parameters."""
        return self._torch_device

    # ------------------------------------------------------------------ loading

    def _repo(self, name: str) -> str:
        """Where a component's weights live: this repo's `name/` folder when bundled,
        its upstream checkpoint otherwise.

        Bundled weights come down one component at a time, when it is first used, so an
        OCR-only caller never downloads the 28 GB brain. A local checkout is read in place.
        Joining the folder onto a Hub id ("org/repo/brain") only ever worked locally.
        """
        if not self.config.bundle_weights:
            return self.config.component(name)
        root = self.name_or_path or ""
        if os.path.isdir(root):
            return os.path.join(root, name)
        from huggingface_hub import snapshot_download

        local = snapshot_download(
            root,
            allow_patterns=[f"{name}/*"],
            token=getattr(self, "_hub_token", None) or _hf_token(),
            revision=getattr(self.config, "_commit_hash", None),
        )
        return os.path.join(local, name)

    def _component(self, name: str) -> Any:
        cached = self._cache.get(name)
        if cached is not None:
            return cached
        with self._cache_lock:
            # Re-check inside the lock: another thread may have loaded it while we waited.
            if name not in self._cache:
                self._cache[name] = getattr(self, f"_load_{name}")()
            return self._cache[name]

    def _load_brain(self):
        """The brain, image-capable: it grounds boxes and reads attached images, which
        a text-only auto class would reject.

        Its FP8 weights are dequantized as they load. Run through transformers' FP8
        kernels they computed nonsense -- a receipt question was answered in looping
        tokens from four scripts -- where the same weights serve correctly on vLLM.
        Dequantized, they compute in bf16: exact, for about 54 GB of memory.
        """
        from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

        repo = self._repo("brain")
        kwargs = {}
        quant = getattr(AutoConfig.from_pretrained(repo, token=_hf_token()), "quantization_config", None)
        if isinstance(quant, dict) and quant.get("quant_method") == "fp8":
            from transformers import FineGrainedFP8Config

            kwargs["quantization_config"] = FineGrainedFP8Config(
                activation_scheme=quant.get("activation_scheme", "dynamic"),
                weight_block_size=tuple(quant.get("weight_block_size") or (128, 128)),
                dequantize=True,
            )
        return {
            "model": AutoModelForImageTextToText.from_pretrained(
                repo,
                dtype=self._dtype,
                device_map=self.config.device_map,
                token=_hf_token(),
                **kwargs,
            ).eval(),
            "processor": AutoProcessor.from_pretrained(repo, token=_hf_token()),
        }

    def _load_segmenter(self):
        """the segmenter, box-prompted.

        The pip loader is patched to load with assign=True: under torch 2.5+ the plain
        path leaves meta tensors and the predictor produces empty masks. The
        detection service carries the same patch.
        """
        import logging

        import sam2.build_sam as build_sam
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        original = build_sam._load_checkpoint

        def patched(model, ckpt_path):
            if ckpt_path is not None:
                state = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
                model.load_state_dict(state, assign=True)
                logging.info("loaded the segmenter checkpoint (assign=True)")

        build_sam._load_checkpoint = patched
        try:
            if self.config.bundle_weights:
                # Bundled: the architecture file and checkpoint both sit in this repo,
                # and are built the way build_sam2 builds them from the package's own.
                from hydra.utils import instantiate
                from omegaconf import OmegaConf

                folder = self._repo("segmenter")
                cfg = OmegaConf.load(os.path.join(folder, "config.yaml"))
                OmegaConf.resolve(cfg)
                model = instantiate(cfg.model, _recursive_=True)
                patched(model, os.path.join(folder, "model.pt"))
                predictor = SAM2ImagePredictor(model.to(self._torch_device).eval())
            else:
                predictor = SAM2ImagePredictor.from_pretrained(
                    self.config.component("segmenter"), device=str(self._torch_device)
                )
        finally:
            build_sam._load_checkpoint = original

        if torch.cuda.is_available():
            predictor.model = predictor.model.half()
        return predictor

    def _load_ocr_vlm(self):
        """Load the document VLM with an image-capable auto class.

        AutoModelForCausalLM resolves to a text-only wrapper for these checkpoints. Its
        forward() then rejects `pixel_values` and `mm_token_type_ids`, and never sets
        `cache_position`, which the custom modeling code indexes into -- the two
        unrelated-looking errors we chased. AutoModelForImageTextToText is the correct
        class for the document reader and an alternative document model alike.
        """
        from transformers import AutoProcessor

        try:
            from transformers import AutoModelForImageTextToText as AutoVLM
        except ImportError:  # older transformers
            from transformers import AutoModelForVision2Seq as AutoVLM

        repo = self._repo("ocr_vlm")
        token = _hf_token()
        return {
            "model": AutoVLM.from_pretrained(
                repo,
                torch_dtype=self._dtype,
                trust_remote_code=True,
                token=token,
                # Explicit, so a transformers default change cannot silently reintroduce
                # a backend this checkpoint does not survive.
                attn_implementation=os.environ.get("OCR_ATTN_IMPL", "sdpa"),
            )
            .to(self._torch_device)
            .eval(),
            "processor": AutoProcessor.from_pretrained(repo, trust_remote_code=True, token=token),
        }

    def _line_detector_settings(self) -> dict:
        """How each worker builds its line detector: detection + recognition, on CPU.

        The GPU build conflicts with torch/vLLM's CUDA runtime, and detection is cheap
        enough on CPU that this costs about a second.
        """
        settings = {
            "lang": "en",
            "device": "cpu",
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            # Several workers run at once; each takes its share of the cores instead
            # of all of them, which oversubscribed the CPU.
            "cpu_threads": max(1, (os.cpu_count() or 8) // max(1, self.config.line_detector_pool_size)),
        }
        if self.config.bundle_weights:
            # Read from this repo, each under the name saved with it: given only a
            # folder, PaddleOCR assumes its own default recogniser and refuses the one it
            # finds ("Model name mismatch").
            detector, recognizer = self._repo("line_detector"), self._repo("line_recognizer")
            settings |= {
                "text_detection_model_name": _paddle_name(detector),
                "text_detection_model_dir": detector,
                "text_recognition_model_name": _paddle_name(recognizer),
                "text_recognition_model_dir": recognizer,
            }
        return settings

    def _layout_settings(self) -> dict | None:
        """How each worker builds its layout detector, or None when there is none.

        Classifying a page into titles, paragraphs, tables and figures is a closed-set
        detection problem: a small CPU detector answers it in milliseconds where a
        structured generation costs a full decode.
        """
        if self.config.bundle_weights:
            folder = self._repo("layout")
            return {"model_name": _paddle_name(folder), "model_dir": folder}
        name = self.config.components.get("layout")
        return {"model_name": name} if name else None

    def _load_asr(self):
        # the fast recogniser's toolkit, not transformers. Kept behind the same accessor so callers cannot tell.
        # The toolkit is a heavy optional dependency; when it is absent every request
        # routes to the speech recogniser instead of failing, which is a real degradation but a
        # working one.
        # Optional, and imported inside `try` so trust_remote_code's import check -- which
        # demands every package the file names -- does not make it a requirement.
        try:
            import nemo.collections.asr as nemo_asr
        except ImportError as exc:
            raise RuntimeError("ENABLE_FAST_ASR needs nemo_toolkit[asr] installed") from exc

        return nemo_asr.models.ASRModel.from_pretrained(self.config.component("asr"))

    def _load_asr_fallback(self):
        """the speech recogniser, placed by accelerate rather than moved afterwards.

        the speech recogniser ties `proj_out` to the decoder embeddings, so it is absent from the
        checkpoint and transformers newly-initialises it -- on the meta device. A
        later `.to(device)` then dies on that one tensor with "Cannot copy out of meta
        tensor; no data!", which the service reported as a bare 501 and which made
        every transcription fail. It depends on load order, so it reproduces
        intermittently and reads as flakiness. `device_map` materialises tied and
        untied parameters alike, and never calls .to() on a meta tensor.
        """
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        repo = self._repo("asr_fallback")
        token = _hf_token()
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            repo, dtype=self._dtype, device_map=str(self._torch_device), token=token
        )
        model.tie_weights()
        processor = AutoProcessor.from_pretrained(repo, token=token)

        return pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            # No chunk_length_s: audio past 30 s goes through the recogniser's own
            # sequential long-form decoding, window by window, which is what makes the
            # repetition guards in _RECOGNISER_GUARDS available and bounds memory per
            # window. Pipeline chunking stitched overlapping windows instead: on a
            # 104 s support call it looped "5-5-5" through a card number, glued words
            # at the seams ("websitesince"), and broke word alignment outright.
            # Measured on the H100: segments 2.6 s / 1.9 GiB peak, words 10.2 s / 4.8 GiB.
            batch_size=1,
        )

    def _load_diarizer(self):
        # the diarization library Pipeline -- not a PreTrainedModel, so it is wrapped rather than
        # subclassed. Weights are gated: an anonymous fetch returns 401, so HF_TOKEN must
        # be set in the environment or passed to from_pretrained.
        from pyannote.audio import Pipeline

        pipeline = Pipeline.from_pretrained(self._repo("diarizer"), token=_hf_token())
        return pipeline.to(self._torch_device)

    def _load_guard(self):
        """The text guard: a safety classifier answering "safe", or "unsafe" and the
        violated categories, prompted with the S1-S14 taxonomy interfaze's guard uses."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        repo = self._repo("guard")
        token = _hf_token()
        return {
            "model": AutoModelForCausalLM.from_pretrained(
                repo, dtype=self._dtype, device_map=str(self._torch_device), token=token
            ).eval(),
            "tokenizer": AutoTokenizer.from_pretrained(repo, token=token),
        }

    def _load_forecaster(self):
        """The time-series model, compiled exactly as JigsawStack's forecast service does:
        1,024 points of context, a 256-step horizon, flip-invariant, non-negative when
        the history is."""
        import timesfm

        # Built in float32 whatever the process default is. TimesFM creates its layers at
        # the default dtype and feeds them float32, and another component's loader left
        # the default at bf16: the forecast then failed with "expected mat1 and mat2 to
        # have the same dtype" depending only on which model happened to load first.
        previous = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        try:
            model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                self._repo("forecaster"), torch_compile=False, device=str(self._torch_device)
            )
        finally:
            torch.set_default_dtype(previous)
        model.model.float()
        model.compile(
            timesfm.ForecastConfig(
                max_context=MAX_CONTEXT,
                max_horizon=MAX_HORIZON,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )
        return model

    # ---------------------------------------------------------------- guardrails

    # Longer text is classified by its last this-many characters: a prompt of tens of
    # thousands of tokens is prefilled in one pass here, on a card other models share.
    _GUARD_MAX_CHARS = 64_000

    @torch.inference_mode()
    def moderate(self, text: str) -> dict:
        """The guard's verdict on one user message: `output` is its raw answer, "safe" or
        "unsafe" followed by a comma-separated list of codes on the next line."""
        guard = self._component("guard")
        tokenizer, model = guard["tokenizer"], guard["model"]
        conversation = [
            {"role": "user", "content": [{"type": "text", "text": (text or "")[-self._GUARD_MAX_CHARS :]}]}
        ]
        encoded = tokenizer.apply_chat_template(
            conversation,
            categories=GUARD_CATEGORIES,
            excluded_category_keys=[],
            return_tensors="pt",
            return_dict=True,
        )
        inputs = {k: v.to(model.device) for k, v in encoded.items()}
        prompt_len = inputs["input_ids"].shape[1]
        out = model.generate(
            **inputs, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.eos_token_id,
            output_scores=True, return_dict_in_generate=True,
        )
        generated = out.sequences[0, prompt_len:]
        answer = tokenizer.decode(generated, skip_special_tokens=True).strip()

        # Unsafe only when the guard is sure. Its verdict is the likelier of "safe" and
        # "unsafe", and plain file requests -- "Extract all text", "Transcribe this audio
        # file" -- came out unsafe S8 at 0.59-0.73, blocking OCR under a guard. Every real
        # violation tested scored 0.986 or more, every safe prompt 0.014 or less; thresholding
        # this probability is how the guard is meant to be calibrated.
        safe, unsafe = (tokenizer.encode(w, add_special_tokens=False)[0] for w in ("safe", "unsafe"))
        p_unsafe = None
        for step, token in enumerate(generated.tolist()):
            if token in (safe, unsafe):
                probs = torch.softmax(out.scores[step][0].float(), dim=-1)
                p_unsafe = float(probs[unsafe] / (probs[unsafe] + probs[safe]))
                break
        if answer.startswith("unsafe") and p_unsafe is not None and p_unsafe < _GUARD_UNSAFE_THRESHOLD:
            answer = "safe"
        return {
            "output": answer,
            "unsafe_probability": p_unsafe,
            "prompt_tokens": int(prompt_len),
            "completion_tokens": int(generated.shape[0]),
        }

    # ---------------------------------------------------------------- forecasting

    @torch.inference_mode()
    def forecast(self, y: dict, horizon: int) -> dict:
        """The next `horizon` values of a date -> value series, oldest first.

        JigsawStack's service, step for step: dates parsed and sorted, the step inferred
        from them, the horizon clamped to the compiled maximum, the last 1,024 points as
        context, and the forecast rounded to integers when every input was one.
        """
        import numpy as np

        integers = forecasting.all_integers(list(y.values()))
        dts, values = forecasting.parse_dates_sorted(y)
        if len(values) < forecasting.MIN_FORECAST_POINTS:
            raise ValueError("At least 5 unique dates are required")
        step = forecasting.infer_step(dts)
        horizon = max(1, min(int(horizon), MAX_HORIZON))
        series = np.asarray(values, dtype=np.float32)[-MAX_CONTEXT:]
        try:
            point, _ = self._component("forecaster").forecast(horizon=horizon, inputs=[series])
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        preds = point[0].tolist()
        preds = (
            np.rint(np.asarray(preds, dtype=np.float64)).astype(np.int64).tolist()
            if integers
            else [round(float(p), 4) for p in preds]
        )
        return {"timestamp": forecasting.future_timestamps(dts[-1], horizon, step), "value": preds}

    # ---------------------------------------------------------------- detection

    @torch.inference_mode()
    def detect(self, image, prompts: Sequence[str], *, return_masks: bool = False) -> dict:
        """Open-vocabulary detection, as the service's object_detection tool answers it.

        Boxes and labels come from the brain, asked the shared grounding question
        (`grounding.prompt`); outlines, and masks when asked for, from the segmenter.
        Returns `{"detected_objects": [{label, bounds, polygon?, mask?}], width, height}`
        in the image's own pixels.
        """
        wanted = [p for p in prompts if p and p.strip()]
        source = _load_image(image)
        if not wanted:
            return {"detected_objects": [], "width": source.width, "height": source.height}
        # Photographs are grounded at 1 MP: at 1.48 MP the sword's box landed on the
        # armour's chest and "every object" looped to the token cap.
        img = _fit_pixels(source, self.config.object_ground_max_pixels)
        raw = self._brain_vision(img, grounding.prompt(wanted, "object"), max_new_tokens=GROUND_MAX_TOKENS)
        found = grounding.to_pixels(grounding.parse(raw, ", ".join(wanted)), source.width, source.height)
        masks: list[str] = []
        outlines: list = []
        if found:
            boxes = [
                [
                    e["bounds"]["top_left"]["x"],
                    e["bounds"]["top_left"]["y"],
                    e["bounds"]["bottom_right"]["x"],
                    e["bounds"]["bottom_right"]["y"],
                ]
                for e in found
            ]
            masks, outlines = self._segment_boxes_with_outlines(source, boxes)
        objects = []
        for i, element in enumerate(found):
            entry: dict = {"label": element["label"], "bounds": element["bounds"]}
            if i < len(outlines) and outlines[i]:
                entry["polygon"] = outlines[i]
            if return_masks and i < len(masks):
                entry["mask"] = masks[i]
            objects.append(entry)
        return {"detected_objects": objects, "width": source.width, "height": source.height}

    def _fit_for_ocr(self, image):
        """Load and bound an image exactly as the OCR and detect paths do.

        Shared so a caller that grounded boxes elsewhere segments against the SAME
        pixel grid those boxes were measured on.
        """
        return _fit_pixels(_load_image(image), self.config.ocr_max_pixels)

    def _segment_boxes_with_outlines(
        self, img, boxes: list[list[int]]
    ) -> tuple[list[str], list[list | None]]:
        """Masks and their outlines, from one segmentation pass.

        The outline is traced from the boolean array the segmenter returns, not from
        the PNG: encoding and decoding it again only to find its edge would be work
        for nothing when the array is right here.
        """
        import numpy as np

        predictor = self._component("segmenter")
        with (
            _SEGMENTER,
            torch.inference_mode(),
            torch.autocast("cuda", dtype=torch.float16, enabled=torch.cuda.is_available()),
        ):
            predictor.set_image(np.array(img))
            masks, _scores, _ = predictor.predict(
                box=np.array(boxes, dtype=np.float32), multimask_output=False
            )

        masks = np.asarray(masks)
        if masks.ndim == 4:
            masks = masks[:, 0]
        elif masks.ndim == 2:
            masks = masks[None]
        masks = masks.astype(bool)
        return ([_encode_mask_array(m) for m in masks], [_outline_of(m) for m in masks])

    # ---------------------------------------------------------------------- OCR

    @torch.inference_mode()
    def ocr(self, source, *, return_markdown: bool = False, page_range: list[int] | None = None) -> dict:
        """OCR an image, a PDF, or a .docx.

        PDFs are rendered page by page and the results concatenated into one document,
        matching the published contract: `sections` flattened across pages, `height`
        summed, `width` the widest page, and per-page bounds offset by the cumulative
        height so every box is addressable in one stacked coordinate space.

        .docx already contains text, so it is read rather than rasterised -- running OCR
        over a re-rendered Word file would only add recognition errors to text we
        already have exactly.
        """
        kind = contracts.document_kind(source)
        if kind == "pdf":
            return self._ocr_pdf(source, return_markdown=return_markdown, page_range=page_range)
        if kind == "docx":
            return _ocr_docx(source)
        if kind == "text":
            return _read_text_document(source)
        return self._ocr_image(source, return_markdown=return_markdown)

    def _ocr_pdf(self, source, *, return_markdown: bool, page_range) -> dict:
        import pymupdf  # `fitz` is the deprecated alias

        data = contracts.fetch_bytes(source)
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            total = doc.page_count
            first, last = 1, total
            if page_range and len(page_range) == 2:
                # 1-based inclusive, per the tool schema. Clamp rather than raise: a
                # model asking for pages 1-50 of a 3-page file means "all of it".
                first = max(1, int(page_range[0]))
                last = min(total, int(page_range[1]))
                if first > last:
                    first, last = last, first

            # A 77-page paper OCRs every page at 1-2s each and then blows the request
            # deadline, which is a worse outcome than answering from the first N: the
            # caller gets nothing at all. Cap it and say so.
            capped = min(last, first - 1 + self.config.max_pdf_pages)

            pages, frames = [], []
            for index in range(first - 1, capped):
                page = doc.load_page(index)
                rect = page.rect
                dpi = contracts.pdf_render_dpi(
                    rect.width,
                    rect.height,
                    dpi=self.config.pdf_render_dpi,
                    max_pixels=self.config.ocr_max_pixels,
                    small_pixels=self.config.ocr_upscale_below_pixels,
                )
                pages.append(_pixmap_to_image(page.get_pixmap(dpi=dpi)))
                # Boxes are reported at interfaze's render scale whatever the read used:
                # a client draws them on the page rendered at that scale.
                scale = self.config.pdf_frame_dpi / 72.0
                frames.append((round(rect.width * scale), round(rect.height * scale)))

        # Both stages run across pages concurrently, for different reasons.
        #
        # Line detection runs in the detector worker processes (line_runtime), one
        # engine each, so the pages genuinely overlap. This loop used to be serial,
        # which on a 50-page document was ~2s a page of wall-clock that nothing else
        # was waiting on.
        #
        # The VLM calls are HTTP to a vLLM server with continuous batching, so N pages
        # in flight cost barely more than one.

        fitted = [_fit_pixels(_load_image(page), self.config.ocr_max_pixels) for page in pages]

        # Both stages in flight together, not one phase after the other. They compete
        # for different things -- line detection for CPU, the VLM for a socket and the GPU
        # behind it -- so a barrier between them only adds their durations: 9.0s and
        # 8.5s measured separately on one page, against 9.0s if they overlap.
        #
        # submit(), not map(): map blocks until its whole iterable is done, so
        # scheduling detection with it and then reading is exactly the barrier this
        # is meant to remove. Futures let both queues drain at once.
        detect_workers = max(1, min(self.config.line_detector_pool_size, len(fitted)))
        read_workers = max(1, min(self.config.ocr_page_concurrency, len(fitted)))

        # Layout goes in as one job covering every page, not one job per page. It is
        # the third stage and it is cheap -- a fraction of a second against a VLM read
        # of several seconds -- so it finishes inside the time the other two already
        # take, and fanning it out would only take workers from line detection.
        def classify_pages() -> list[tuple[list[dict], str]]:
            return [self._detect_layout(img) for img in fitted]

        with concurrent.futures.ThreadPoolExecutor(max_workers=detect_workers + read_workers + 1) as pool:
            detect_jobs = [pool.submit(self._detect_text_lines, img) for img in fitted]
            read_jobs = [pool.submit(self._read_document, img) for img in fitted]
            layout_job = pool.submit(classify_pages)
            detector_lines = [job.result() for job in detect_jobs]
            documents = [job.result() for job in read_jobs]
            layouts = layout_job.result()

        results = []
        for img, frame, lines, (regions, markdown), (blocks, status) in zip(
            fitted, frames, detector_lines, documents, layouts, strict=True
        ):
            results.append(
                contracts.rescale_page(
                    self._assemble(
                        img,
                        lines,
                        regions,
                        markdown,
                        return_markdown=return_markdown,
                        layout=blocks,
                        layout_status=status,
                    ),
                    *frame,
                )
            )

        sections: list[dict] = []
        for offset, result in enumerate(results):
            # No coordinate translation. interfaze keeps bounds page-local and stamps
            # each section with its page and that page's frame.
            # Stacking pages onto one canvas -- which this used to do -- silently moved
            # every box on every page after the first, with nothing to signal it.
            sections.extend(
                contracts.backfill_sections(
                    result.get("sections") or [],
                    page=first + offset,
                    width=result.get("width") or 0,
                    height=result.get("height") or 0,
                )
            )

        page_text = "\n\n".join(r["text"] for r in results if r.get("text"))
        return {
            "text": page_text,
            # Each page's text on its own, so the model can be shown where pages start.
            "page_texts": [r.get("text") or "" for r in results],
            "context": page_text if return_markdown else None,
            "sections": sections,
            "has_text": any(r.get("has_text") for r in results),
            # Max width, summed height: the aggregate shape interfaze reports for
            # back-compat. It matches no single page, which is exactly why every
            # section carries its own frame.
            "width": max((r.get("width") or 0) for r in results) if results else 0,
            "height": sum((r.get("height") or 0) for r in results),
            "total_pages": total,
            "pages_processed": [first, last],
            "layout": [
                {**block, "page": offset + first}
                for offset, result in enumerate(results)
                for block in (result.get("layout") or [])
            ],
            # Per page, because a document can lose layout on page 7 alone and an
            # aggregate that reports only "ok" would read as a clean run.
            "layout_status": "; ".join(sorted({str(r.get("layout_status") or "not run") for r in results}))
            if results
            else "no pages",
        }

    def _ocr_image(self, image, *, return_markdown: bool = False) -> dict:
        """Two-stage OCR: a detector supplies geometry and confidence, a VLM supplies
        structure.

        Neither half is sufficient on its own. an alternative document model reconstructs reading order,
        tables and markdown but emits no confidence at all -- its authors dropped mAP
        for exactly that reason. The line detector supplies calibrated per-word scores,
        which `average_confidence` needs so the low-confidence fallback can fire.
        """
        # Bound resolution here, once, so the detector and the VLM see the SAME image
        # and therefore agree on a coordinate space. Capping only inside the VLM path
        # left the line detector chewing a 16.8 MP render on CPU -- 65s instead of 5s -- and the
        # two halves measuring boxes against different pixel grids.
        source = contracts.enhance_scan(_load_image(image))
        img = _fit_pixels(source, self.config.ocr_max_pixels)
        if source.width * source.height < self.config.ocr_upscale_below_pixels:
            # Small type in a small image is a few pixels a glyph, which the reader
            # guesses at; twice the pixels and it reads it. Boxes are mapped back below.
            img = _fit_pixels(_upscale(source, 2), self.config.ocr_max_pixels)

        # Timed separately because the two halves have very different cost profiles and
        # very different fixes: line detection is CPU-bound detection, the VLM is GPU decode.
        # Guessing which dominates sends you optimising the wrong one.
        # The two halves contend for nothing: line detection is CPU-only, the document VLM is
        # an HTTP call to a GPU server, and their outputs stay independent until the
        # stitch. Run one after the other they simply add -- measured 9.0s + 8.5s on a
        # single 0.69 MP page, for 17.5s of a 29s request.
        #
        # What forced the ordering was `_denser_budget`: it needs the detected line
        # heights before it can decide a page deserves more pixels, and that decision
        # changes which image the VLM should read. So both start at the current budget
        # and the second pass is paid for only when a page actually escalates.
        t0 = time.perf_counter()
        detector_lines, regions, vlm_markdown, layout, layout_status = self._read_page(img)
        t_page = time.perf_counter() - t0

        bigger = self._denser_budget(source, img, detector_lines)
        if bigger:
            img = bigger
            t0 = time.perf_counter()
            detector_lines, regions, vlm_markdown, layout, layout_status = self._read_page(img)
            t_page += time.perf_counter() - t0

        # One figure, because the stages overlap now: splitting the wall-clock between
        # them would imply a division that no longer exists.
        print(
            f"[ocr timing] {img.width}x{img.height} "
            f"detect+vlm(parallel)={t_page:.1f}s "
            f"lines={len(detector_lines)} regions={len(regions)}",
            flush=True,
        )
        # In the caller's pixels, not the resized copy's.
        return contracts.rescale_page(
            self._assemble(
                img,
                detector_lines,
                regions,
                vlm_markdown,
                return_markdown=return_markdown,
                t_detect=t_page,
                t_vlm=0.0,
                layout=layout,
                layout_status=layout_status,
            ),
            source.width,
            source.height,
        )

    def _detect_layout(self, img) -> tuple[list[dict], str]:
        """Typed blocks for one page: label, score and bounds in the image's frame.

        Why the result is inspected so defensively: the detector's output wrapper
        differs between releases -- a plain dict, a dict under "res", or an object
        exposing .json -- and an unrecognised shape yields no blocks at all. When that
        happens the reason is returned alongside the blocks and surfaced in the OCR
        payload, because a silently empty layout is indistinguishable from a page that
        genuinely has no regions. Returned rather than stored on the instance: pages are
        classified concurrently, and one page's outcome must not overwrite another's.
        """
        # Bundled, the detector is in this repo; otherwise the deployment names it.
        # Asking the config for a component name alone switched layout off for every
        # bundled download, whose config names none.
        try:
            if self._layout_settings() is None:
                return [], "unavailable: no layout component configured"
        except Exception as exc:
            return [], f"unavailable: {type(exc).__name__}: {exc}"[:200]

        try:
            pages = self._detection_runtime().run(layout_pages, _to_numpy(img)) or []
        except Exception as exc:
            return [], f"predict failed: {type(exc).__name__}: {exc}"[:200]

        if not pages:
            return [], "predict returned no pages"

        blocks: list[dict] = []
        shapes: list[str] = []
        for page in pages:
            data: Any = page
            for accessor in ("res", "json"):
                if isinstance(data, dict) and accessor in data:
                    data = data[accessor]
                elif not isinstance(data, dict) and hasattr(data, accessor):
                    data = getattr(data, accessor)
            if isinstance(data, dict) and "res" in data:
                data = data["res"]

            found: list = []
            if isinstance(data, dict):
                for key in ("boxes", "layout_boxes", "dt_polys", "det_boxes"):
                    if data.get(key):
                        found = data[key]
                        break
                if not found:
                    shapes.append(f"{type(page).__name__}:{sorted(data)[:6]}")
            else:
                shapes.append(type(page).__name__)

            for box in found:
                coords = box.get("coordinate") or box.get("bbox") if isinstance(box, dict) else None
                if not coords or len(coords) != 4:
                    continue
                x1, y1, x2, y2 = (float(v) for v in coords)
                if x2 <= x1 or y2 <= y1:
                    continue
                blocks.append(
                    {
                        "type": str(box.get("label") or "text").lower(),
                        "score": round(float(box.get("score") or 0.0), 4),
                        "bounds": contracts.to_bounds(
                            [x1, y1, x2, y2], src="xyxy", img_w=img.width, img_h=img.height
                        ).as_dict(),
                    }
                )

        blocks.sort(key=lambda b: (b["bounds"]["top_left"]["y"], b["bounds"]["top_left"]["x"]))
        if not blocks and shapes:
            return blocks, f"no boxes in result shape {shapes[0]}"[:200]
        return blocks, "ok"

    def _read_page(self, img):
        """Detect lines, classify layout, and read the document at the same time.

        Three threads: two block on CPU inside line detection, the third on a socket waiting
        for the VLM server, so none holds the GIL for the part that costs. Layout is
        independent of the other two, so it is free wall-clock rather than an extra
        stage.
        """

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            detect = pool.submit(self._detect_text_lines, img)
            read = pool.submit(self._read_document, img)
            layout = pool.submit(self._detect_layout, img)
            return detect.result(), *read.result(), *layout.result()

    def _denser_budget(self, source, img, detector_lines):
        """Re-fit the page at the dense budget when the first read looks pixel-starved.

        Two independent signals, because they catch different documents. Short lines
        mean small type -- a dense paper -- which is where a 1 MP read lost subscripts
        entirely. Low mean confidence means the recogniser is guessing, which is the
        degraded-scan case. Either is worth the extra pixels; neither fires on an
        ordinary page, so the common path keeps the cheaper budget.
        """
        if self.config.ocr_dense_max_pixels <= self.config.ocr_max_pixels:
            return None
        if img.width * img.height >= source.width * source.height:
            return None  # already showing every pixel there is
        if not detector_lines:
            return None

        heights = [line.bounds.height for line in detector_lines if line.bounds.height > 0]
        if not heights:
            return None
        median_height = sorted(heights)[len(heights) // 2]
        small = median_height / img.height < self.config.ocr_small_line_ratio

        scores = [line.average_confidence for line in detector_lines if line.average_confidence is not None]
        weak = bool(scores) and (sum(scores) / len(scores)) < self.config.ocr_retry_confidence

        if not (small or weak):
            return None
        denser = _fit_pixels(source, self.config.ocr_dense_max_pixels)
        if denser.width <= img.width:
            return None
        print(
            f"[ocr] re-reading at {denser.width}x{denser.height} (small_type={small} low_confidence={weak})",
            flush=True,
        )
        return denser

    def _assemble(
        self,
        img,
        detector_lines,
        regions,
        vlm_markdown,
        *,
        return_markdown: bool,
        t_detect: float = 0.0,
        t_vlm: float = 0.0,
        layout: list[dict] | None = None,
        layout_status: str = "not run",
    ) -> dict:
        """Stitch one page's detector geometry to its VLM text."""

        stitched = (
            contracts.stitch(regions, detector_lines)
            if regions
            else contracts.merge_lines_into_sections(detector_lines)
        )

        # ONE SECTION PER PAGE. This is a hard part of the interfaze contract, not a
        # presentation choice: consumers index `sections[0]` as the page, and derive the
        # per-page height as `result.height / sections.length` to map page-local
        # coordinates. Emitting a section per text region made a single page look like
        # three, so every box was scaled by a third of the real page height and
        # attributed to the wrong page.
        sections = [contracts.collapse_to_page(stitched)] if stitched else []

        # Both of these are the document VLM's own output -- that is the property that
        # matters, and the one that was broken when detector text could leak in.
        #
        # `text` is the COMPLETE document. Building it from regions instead looked
        # tidier but silently lost content: region extraction only captures text
        # following a data-bbox attribute, so anything the document reader emits unboxed vanished.
        # Losing an invoice total to make a substring assertion hold is a bad trade.
        region_text = "\n".join(s.text for s in sections if s.text.strip())
        # Returned by the reader rather than stashed on self: pages are OCR'd
        # concurrently, and instance state would let one page's text overwrite another's.
        markdown = vlm_markdown or region_text
        if contracts.describes_a_photo(detector_lines, markdown):
            markdown, region_text, sections = "", "", []
        text = markdown or region_text
        return {
            "_timing": {"detect_s": round(t_detect, 2), "vlm_s": round(t_vlm, 2)},
            # `text` is always the VLM's own output and is the single source of truth
            # for document text. `context` is the markdown rendering of the same thing,
            # returned only when asked for.
            "text": text,
            "context": markdown if return_markdown else None,
            "sections": contracts.backfill_sections(
                [s.as_dict() for s in sections], page=1, width=img.width, height=img.height
            ),
            "has_text": bool(text.strip()),
            # Typed blocks with their own text and bounds, produced by a detector
            # rather than a structured generation. The caller gets the same three
            # things it used to -- what kind of block, where it is, what it says --
            # without a decode per page.
            "layout": contracts.attach_text_to_layout(
                layout or [],
                detector_lines,
                min_containment=self.config.ocr_stitch_min_containment,
                regions=regions or (),
            ),
            "layout_status": layout_status,
            "width": img.width,
            "height": img.height,
            "total_pages": 1,
        }

    def _detect_text_lines(self, img) -> list[OCRLine]:
        """the line detector detection + recognition -> lines with quads and confidence.

        the line detector 3.x renamed the entry point (`.ocr(cls=True)` -> `.predict()`), changed
        the result shape from nested tuples to a dict of parallel lists, and removed
        `show_log`. Both generations are supported because the installed version depends
        on the image build and a hard failure here loses the confidence signal that the
        low-confidence fallback depends on.
        """
        # line detection's C++ detector aborts with a bare `RuntimeError: std::exception` on
        # large inputs, which is what the single 1 MP ingestion cap was really
        # protecting against. But the two stages do not need the same picture: line detection
        # only contributes geometry and confidence, while the VLM has to actually read
        # the glyphs, and starving IT of pixels is what lost subscripts. So detection
        # runs on a downscaled copy and its boxes are scaled back into the caller's
        # frame. The frames must still agree at the end -- that is what the stitch
        # matches on -- so the rescale is arithmetic, not an approximation.
        detect_img = _fit_pixels(img, self.config.line_detector_max_pixels)
        scale_x = img.width / detect_img.width
        scale_y = img.height / detect_img.height

        arr = _to_numpy(detect_img)
        rows = self._detection_runtime().run(line_rows, arr)

        lines: list[OCRLine] = []
        for quad, text, score in rows:
            if not (text or "").strip():
                continue
            if scale_x != 1.0 or scale_y != 1.0:
                quad = [[x * scale_x, y * scale_y] for x, y in quad]
            bounds = contracts.to_bounds(quad, src="quad", img_w=img.width, img_h=img.height)
            lines.append(
                OCRLine(
                    text=text,
                    bounds=bounds,
                    average_confidence=score,
                    words=contracts.split_words(text, bounds, score),
                )
            )
        return lines

    def _read_document(self, img) -> tuple[list[contracts.VLMRegion], str]:
        """Run the configured document VLM: regions in reading order, plus markdown.

        Two families are supported because they speak different output languages:
        the document reader emits HTML whose elements carry `data-bbox`, an alternative document model emits a JSON list
        of (bbox, category, text). Both are reduced to the same VLMRegion here so the
        stitching step downstream does not care which one ran.
        """
        # Chosen by configuration, not by sniffing the component's name. Matching a
        # substring of the repo id meant renaming anything silently rerouted OCR to
        # the other reader, which then failed on the first page it was handed.
        if self.config.ocr_vlm_format == "html":
            return self._read_document_html(img)
        return self._read_document_json(img), ""

    def _read_document_html(self, img) -> tuple[list[contracts.VLMRegion], str]:
        """the document reader -> HTML with data-bbox attributes.

        The prompt comes from the upstream `the document reader-ocr` package when it is installed,
        because the document reader is fine-tuned against one specific instruction and paraphrasing
        it degrades output. The literal is a fallback, not an alternative.
        """
        try:
            from chandra.prompts import PROMPT_MAPPING

            prompt = PROMPT_MAPPING["ocr_layout"]
        except (ImportError, KeyError):
            prompt = (
                "Convert this document to HTML. Preserve reading order and layout. "
                "Give every block a data-bbox attribute."
            )

        # Served by vLLM when a URL is configured. That is the difference between ~15
        # tok/s through transformers.generate() -- a bare autoregressive loop -- and
        # continuous batching with paged attention and CUDA graphs. On this workload it
        # is the single largest latency lever, and it also removes the 4x run-to-run
        # variance that comes from having no scheduler.
        if self.config.ocr_vlm_url:
            html = self._vlm_via_server(img, prompt)
            return contracts.parse_vlm_regions(html, img), contracts.vlm_markdown(html)

        ocr = self._component("ocr_vlm")
        model, processor = ocr["model"], ocr["processor"]

        # Already bounded by ocr() at ingestion; only the output budget is set here.
        budget = _token_budget(img, self.config.ocr_max_new_tokens)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], return_tensors="pt").to(self._torch_device)
        out = _generate(model, inputs, max_new_tokens=budget, eos_token_id=_stop_ids(model, processor))
        html = processor.decode(out[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)

        return contracts.parse_vlm_regions(html, img), contracts.vlm_markdown(html)

    def _vlm_via_server(self, img, prompt: str) -> str:
        """One chat call to the co-located document reader's vLLM server."""
        import base64 as _b64
        import io as _io

        import httpx

        buf = _io.BytesIO()
        img.save(buf, format="PNG")
        data_url = "data:image/png;base64," + _b64.b64encode(buf.getvalue()).decode()

        resp = httpx.post(
            f"{self.config.ocr_vlm_url.rstrip('/')}/v1/chat/completions",
            json={
                "model": self.config.ocr_vlm_served_name,
                "temperature": 0.0,
                "max_tokens": _token_budget(img, self.config.ocr_max_new_tokens),
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_url}},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            },
            timeout=600,
        )
        resp.raise_for_status()
        return (resp.json()["choices"][0]["message"]["content"] or "").strip()

    def _read_document_json(self, img) -> list[contracts.VLMRegion]:
        """an alternative document model -> (bbox, category, text) tuples in reading order. No confidence."""
        ocr = self._component("ocr_vlm")
        model, processor = ocr["model"], ocr["processor"]
        prompt = (
            "Extract the layout as JSON: a list of "
            '{"bbox":[x1,y1,x2,y2],"category":...,"text":...} in reading order.'
        )
        inputs = processor(images=img, text=prompt, return_tensors="pt").to(self._torch_device)

        # The processor can emit keys the model's generate() rejects, and which keys
        # depends on the transformers/checkpoint pairing. Filtering against forward()'s
        # signature looked tidy but strips arguments generate() manages itself (it left
        # cache_position unset, which the custom modeling code then indexed into).
        #
        # Instead: try the full input, and if transformers names the unused keys in its
        # error, drop exactly those and retry once. Precise, and version-proof.
        out = _generate(model, inputs, max_new_tokens=8192)
        return _parse_layout(processor.decode(out[0], skip_special_tokens=True), img)

    # -------------------------------------------------------------------- audio

    # The only inference path here that was missing this, and the one that could least
    # afford to be: the recogniser's word-timestamp mode keeps per-layer cross-attention
    # tensors, so without inference_mode they stay reachable after the call returns.
    # Perception's live allocation climbed 15.2 -> 17.1 -> 18.2 GiB across successive
    # requests on a card with nothing left to give, and transcription OOMed on 2 MiB.
    @torch.inference_mode()
    def transcribe(
        self, audio: str, *, by_speaker: bool = False, language: str = "auto", word_timestamps: bool = False,
        words_for_speakers: bool = False,
    ) -> dict:
        """Transcribe, and optionally attribute each word to a speaker.

        Language selection routes between two models rather than picking one compromise:
        the fast recogniser covers 25 languages with better English WER and native word timestamps;
        the speech recogniser covers 99 and takes everything else. Their multilingual averages are the
        same, so this is a coverage decision, not an accuracy one.
        """
        requested = (language or "auto").lower()
        use_fast_asr = _fast_asr_enabled() and requested in self.config.asr_languages
        # Words cost the recogniser four times what segments do, so they are decoded only
        # when something reads them: the caller, or the speaker join. For the join alone,
        # a long recording's are estimated (`asr_exact_words_up_to_s`).
        need_words = by_speaker or word_timestamps or words_for_speakers
        estimate_after_s = (None if word_timestamps
                            else getattr(self.config, "asr_exact_words_up_to_s", None))

        if use_fast_asr:
            words, detected = self._transcribe_fast(audio), requested
            units = words if need_words else _segments(words)
        else:
            units, detected = self._transcribe_long_form(audio, requested, words=need_words,
                                                         estimate_after_s=estimate_after_s)

        result: dict[str, Any] = {
            "text": " ".join(u.text for u in units),
            # What the model actually decoded, not the caller's "auto" echoed back.
            "language_detected": {"code": _language_code(detected) or requested, "confidence": 1.0},
        }

        # Always timed, as interfaze returns them: segments by default, words when asked.
        # This returned [] unless words were requested, which dropped timing the
        # recogniser had already computed.
        if by_speaker:
            turns = self.diarize(audio)
            result["chunks"] = contracts.group_by_speaker(
                contracts.attribute_speakers(units, turns, fill_nearest=True)
            )
        else:
            result["chunks"] = [u.as_dict() for u in units]

        return result

    def diarize(self, audio: str) -> list[contracts.SpeakerTurn]:
        """Who spoke when. Returns turns only -- no words. Join with `attribute_speakers`."""
        annotation = self._component("diarizer")(_waveform(audio))
        return [
            contracts.SpeakerTurn(speaker=str(label), start=float(seg.start), end=float(seg.end))
            for seg, _, label in contracts.speaker_tracks(annotation)
        ]

    def _transcribe_fast(self, audio: str) -> list[contracts.Word]:
        try:
            model = self._component("asr")
            out = model.transcribe([audio], timestamps=True)[0]
        except Exception as exc:
            # the fast recogniser's toolkit imports torchaudio.io, removed in torchaudio 2.9 and replaced by
            # TorchCodec; it also cannot fetch remote audio through lhotse here. Any
            # failure degrades to the speech recogniser, which covers every language the fast recogniser does.
            print(
                f"[asr] the fast recogniser unavailable ({type(exc).__name__}: {exc}); "
                f"falling back to the speech recogniser",
                flush=True,
            )
            return self._transcribe_long_form(audio, words=True)[0]
        return [
            contracts.Word(text=w["word"], start=float(w["start"]), end=float(w["end"]))
            for w in out.timestamp["word"]
        ]

    def _transcribe_long_form(
        self, audio: str, language: str | None = None, *, words: bool = False,
        estimate_after_s: float | None = None,
    ) -> tuple[list[contracts.Word], str | None]:
        """Timed spans -- words, or segments -- plus the language actually used.

        `language` was previously accepted by the tool, forwarded through the service
        and then dropped here, so asking for a specific language did nothing at all.
        the speech recogniser takes it as a decoder prompt; given none, it detects one, and
        `return_language` is how that detection is recovered instead of reporting the
        caller's "auto" back to them as though it were a result.
        """
        from transformers.pipelines.audio_utils import ffmpeg_read

        asr = _one_at_a_time(self._component("asr_fallback"))
        samples = ffmpeg_read(contracts.fetch_bytes(audio), _ASR_RATE)
        if len(samples) > self.config.asr_window_after_s * _ASR_RATE:
            estimate = words and estimate_after_s is not None and len(samples) > estimate_after_s * _ASR_RATE
            if estimate:
                print(f"[asr] {len(samples) / _ASR_RATE / 60:.0f} min: word timing estimated "
                      f"from segments", flush=True)
            units, detected = _transcribe_windows(
                asr,
                samples,
                language,
                words=words and not estimate,
                batch_size=self.config.asr_batch_size,
            )
            return (contracts.estimate_words(units) if estimate else units), detected

        generate = dict(_RECOGNISER_GUARDS)
        if language and language != "auto":
            generate["language"] = language
        call: dict = {
            "return_language": True,
            "return_timestamps": "word" if words else True,
            "generate_kwargs": generate,
        }

        try:
            out = asr(audio, **call)
        except (TypeError, ValueError):
            # Older transformers reject return_language, and a bad language code is
            # the caller's mistake rather than a reason to return nothing.
            out = asr(audio, return_timestamps=True)
        except IndexError:
            # Word-level timestamps are collated inside transformers, and that
            # collation indexes past the end of its own list on some generations
            # (the recogniser's tokenizer, `_decode_asr`). It took the whole capability down:
            # the caller got a 500 and was told transcription was unavailable, when
            # the audio had in fact been recognised and only the per-word alignment
            # failed. Segment timestamps come from a different path, so the text and
            # coarse timing survive; callers that wanted word timing get chunks that
            # are longer than a word rather than nothing at all.
            print("[asr] word timestamps failed to collate; using segment timing", flush=True)
            out = asr(audio, **{**call, "return_timestamps": True})

        chunks = out.get("chunks", [])
        detected = next((c.get("language") for c in chunks if c.get("language")), out.get("language"))
        words = [
            contracts.Word(
                text=chunk["text"].strip(),
                start=float(chunk["timestamp"][0]),
                end=float(
                    chunk["timestamp"][1] if chunk["timestamp"][1] is not None else chunk["timestamp"][0]
                ),
            )
            for chunk in chunks
        ]
        return words, (language if language and language != "auto" else detected)

    # ------------------------------------------------------------------ the brain

    @torch.inference_mode()
    @torch.inference_mode()
    def ground(self, image, prompts: Sequence[str] | None = None) -> dict:
        """Locate UI elements in a screenshot, as the service's gui_detection tool does.

        Runs on the brain rather than a GUI specialist: it grounds well enough on
        ScreenSpot-v2 that a second model would cost VRAM without a matching gain. With
        no prompts it finds every interactive element. Returns
        `{"gui_elements": [{type, bounds}], width, height}` in the image's own pixels,
        each element typed by the phrase that found it, as interfaze types them.
        """
        wanted = [p for p in (prompts or []) if p and p.strip()]
        everything = not wanted
        source = _load_image(image)
        img = _fit_pixels(source, self.config.ground_max_pixels)
        elements = []
        for phrase in wanted or ["interactive element (buttons, links, inputs, icons)"]:
            raw = self._brain_vision(
                img, grounding.prompt([phrase], "ui"), max_new_tokens=GROUND_MAX_TOKENS_UI
            )
            found = grounding.to_pixels(grounding.parse(raw, phrase), source.width, source.height)
            elements += [{"type": "icon" if everything else phrase, "bounds": e["bounds"]} for e in found]
        return {"gui_elements": elements, "width": source.width, "height": source.height}

    def _brain_vision(self, img, prompt: str, max_new_tokens: int = 4096) -> str:
        return self._brain_generate(
            [
                {
                    "role": "user",
                    "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}],
                }
            ],
            max_new_tokens=max_new_tokens,
        )

    def _brain_generate(
        self,
        messages: list[dict],
        *,
        max_new_tokens: int,
        tools: list[dict] | None = None,
        keep_markup: bool = False,
    ) -> str:
        """One brain turn through transformers, with thinking off, as the service runs
        every tool turn and grounding call. Images ride in the messages as PIL images.

        `keep_markup` keeps the tool-call tags, which decoding would otherwise strip."""
        brain = self._component("brain")
        model, processor = brain["model"], brain["processor"]
        images = [
            part["image"]
            for m in messages
            if isinstance(m.get("content"), list)
            for part in m["content"]
            if part.get("type") == "image"
        ]
        text = processor.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = processor(text=[text], images=images or None, return_tensors="pt").to(self._torch_device)
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=_stop_ids(model, processor)
        )
        generated = out[0][inputs["input_ids"].shape[1] :]
        if not keep_markup:
            return processor.decode(generated, skip_special_tokens=True).strip()
        text = processor.decode(generated, skip_special_tokens=False)
        return re.sub(r"<\|(?:im_end|endoftext)\|>", "", text).strip()

    def chat(
        self, messages: list[dict], *, files: Sequence[str] | None = None, max_new_tokens: int = 4096
    ) -> dict:
        """Answer a conversation, reading `files` with OCR, speech, detection and GUI
        grounding when it needs them.

        The service's tool loop, run in-process (`agent.run`): same prompts, same tools,
        same result shapes. Returns `{"content", "precontext": [{name, result}]}` -- prose
        for a person, each tool's full result for a program. Files are paths or URLs,
        referred to as ref-0, ref-1, ... in order.
        """
        return run_agent(
            messages,
            list(files or []),
            generate=lambda convo, tools=None, keep_markup=False: self._brain_generate(
                convo, max_new_tokens=max_new_tokens, tools=tools, keep_markup=keep_markup
            ),
            tools={
                "ocr": self._tool_ocr,
                "stt": self._tool_stt,
                "object_detection": self._tool_detect,
                "gui_detection": self._tool_ground,
            },
            show_images=self._show_images,
        )

    # Each returns (what the caller gets, what the model reads), as the service's tools do.
    def _tool_ocr(self, path: str, args: dict) -> tuple[dict, dict]:
        result = self.ocr(path, page_range=args.get("page_range"))
        full = {
            "extracted_text": result.get("text") or result.get("context") or "",
            "sections": result.get("sections") or [],
            "width": result.get("width"),
            "height": result.get("height"),
            **({"total_pages": result.get("total_pages")} if result.get("pages_processed") else {}),
        }
        # Pages marked, so the model knows what it has read and does not ask again.
        pages = result.get("page_texts") or []
        first = (result.get("pages_processed") or [1])[0]
        text = (
            "\n\n".join(f"--- page {first + i} ---\n{t}" for i, t in enumerate(pages))
            if len(pages) > 1
            else full["extracted_text"]
        )
        return full, {
            "extracted_text": text,
            "width": full["width"],
            "height": full["height"],
            "total_pages": result.get("total_pages"),
        }

    def _tool_stt(self, path: str, args: dict) -> tuple[dict, dict]:
        result = self.transcribe(path, by_speaker=bool(args.get("by_speaker")))
        return result, result

    def _tool_detect(self, path: str, args: dict) -> tuple[dict, dict]:
        result = self.detect(path, args.get("prompts") or [])
        return result, {
            "detected_objects": [
                {k: o[k] for k in ("label", "bounds") if k in o} for o in result["detected_objects"]
            ]
        }

    def _tool_ground(self, path: str, args: dict) -> tuple[dict, dict]:
        result = self.ground(path, args.get("prompts"))
        return result, result

    def _show_images(self, convo: list[dict], refs: dict[str, str]) -> list[dict]:
        """The caller's images, put beside the conversation once a tool has read them.

        Withheld until then: shown an image first, the model transcribes it by eye and
        invents text. With the reading in hand, the image carries what the text cannot
        -- the highlighted item on a receipt, a colour, a crossed-out line.
        """
        images = []
        for path in refs.values():
            try:
                if contracts.document_kind(path) != "image":
                    continue
                images.append(_fit_pixels(_load_image(path), self.config.object_ground_max_pixels))
            except Exception:
                continue
            if len(images) == 4:
                break
        users = [i for i, m in enumerate(convo) if m.get("role") == "user" and m.get("content") != NUDGE]
        if not images or not users:
            return convo
        message = convo[users[-1]]
        content = (
            message["content"]
            if isinstance(message["content"], list)
            else [{"type": "text", "text": message["content"]}]
        )
        shown = list(convo)
        shown[users[-1]] = {
            **message,
            "content": [*content, *({"type": "image", "image": image} for image in images)],
        }
        return shown


# ----------------------------------------------------------------------- helpers


# Keys a vision-language generate() genuinely needs. If transformers reports one of
# these as "not used by the model", the model was loaded with the wrong auto class --
# dropping it would silently discard the image and produce confident nonsense, so the
# error is raised instead.
_ESSENTIAL_INPUTS = frozenset(
    {
        "input_ids",
        "attention_mask",
        "pixel_values",
        "pixel_values_videos",
        "image_grid_thw",
        "video_grid_thw",
    }
)


def _stop_ids(model, processor) -> list[int]:
    """The ids a turn ends on: the checkpoint's own, plus <|im_end|>.

    These checkpoints end a turn with <|im_end|> but list only <|endoftext|> in their
    generation config. A serving engine stops at both; generate() stopped at neither, so
    one page was read on to the 8192-token cap -- 140 regions instead of 16, in five
    minutes.
    """
    ids = model.generation_config.eos_token_id
    ids = [ids] if isinstance(ids, int) else list(ids or [])
    im_end = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end >= 0 and im_end not in ids:
        ids.append(im_end)
    return ids


def _generate(model, inputs, *, max_new_tokens: int, eos_token_id: list[int] | None = None):
    """generate() that tolerates surplus processor keys but never drops the image.

    Processors emit auxiliary tensors that some model versions do not accept, and
    transformers raises rather than ignoring them. Those are safe to drop. Anything in
    _ESSENTIAL_INPUTS is not.
    """
    kwargs = dict(inputs)
    stop = {"eos_token_id": eos_token_id} if eos_token_id else {}
    for _ in range(3):
        try:
            return model.generate(**kwargs, **stop, max_new_tokens=max_new_tokens, do_sample=False)
        except ValueError as exc:
            if "not used by the model" not in str(exc):
                raise
            named = set(re.findall(r"'([A-Za-z_][A-Za-z0-9_]*)'", str(exc)))
            essential = named & _ESSENTIAL_INPUTS
            if essential:
                raise ValueError(
                    f"{model.__class__.__name__} rejected essential input(s) "
                    f"{sorted(essential)} -- this checkpoint needs an image-capable auto "
                    f"class (AutoModelForImageTextToText), not a text-only one."
                ) from exc
            droppable = [k for k in named if k in kwargs]
            if not droppable:
                raise
            for key in droppable:
                kwargs.pop(key)
    raise RuntimeError("generate() kept rejecting its own processor output")


_ASR_RATE = 16_000

# One recogniser call on the GPU at a time, across requests. A word-timed pass peaks near
# 5 GB; six concurrent transcriptions took the card to 81 MiB free and the brain's vLLM
# engine, sharing it, died of the OOM. Taken per call rather than per transcription: held
# for a whole recording, one 95-minute file kept every other transcription waiting ~90 s.
_RECOGNISER = threading.Lock()

# The guard's probability of "unsafe" a verdict needs to block (see `moderate`).
_GUARD_UNSAFE_THRESHOLD = 0.8

# The segmenter keeps the image it was given between set_image and predict. Two requests
# interleaving there cut one image's masks from the other's embedding. Held for that pair
# only -- about a tenth of a second -- not for the request around it.
_SEGMENTER = threading.Lock()


def _one_at_a_time(recogniser):
    def call(*args, **kwargs):
        with _RECOGNISER:
            return recogniser(*args, **kwargs)

    return call


def _loops(text: str, threshold: float = 2.4) -> bool:
    """Text that compresses like a repetition loop, by the reference decoder's own test."""
    import zlib

    raw = text.encode("utf-8")
    return len(raw) > 0 and len(raw) / len(zlib.compress(raw)) > threshold


def _transcribe_windows(
    asr, samples, language: str | None, *, words: bool, batch_size: int
) -> tuple[list[contracts.Word], str | None]:
    """Long audio decoded as interfaze's recogniser decodes it: in windows, in batches.

    Cut at quiet points into windows of at most 30 s (`contracts.speech_windows`), each
    is one short-form decode, and `batch_size` of them run at once. The language is
    detected once, on the first window, as the sequential decoder detects it -- and
    because transformers 5.15 cannot split a batched result that carries languages.
    """
    windows = contracts.speech_windows(samples, _ASR_RATE)
    detected = language if language and language != "auto" else None
    if not windows:
        return [], detected
    generate = dict(_RECOGNISER_GUARDS)
    if detected is None:
        s, e = windows[0]
        probe = asr(
            {"raw": samples[s:e], "sampling_rate": _ASR_RATE},
            return_language=True,
            return_timestamps=True,
            generate_kwargs=dict(generate),
        )
        detected = next(
            (c.get("language") for c in probe.get("chunks", []) if c.get("language")), probe.get("language")
        )
    if detected:
        generate["language"] = detected
    greedy = {"language": detected} if detected else {}

    def window(i):
        # Built per call: the pipeline consumes each input's "raw" as it reads it.
        s, e = windows[i]
        return {"raw": samples[s:e], "sampling_rate": _ASR_RATE}

    def decode(timestamps):
        """Every window, a batch at a time, halving the batch while it does not fit.

        The card is shared, so what fits changes with everything else running. A 95-minute
        recording with word timing ran out of memory at 16 windows a batch and returned a
        500. Once halved, the rest of the recording keeps the smaller size: started again
        at 16 for every batch, it spent each one failing its way down to 4.
        """
        # One batch per call, so another request waits for a batch, not the recording.
        outs, size, first = [], batch_size, 0
        while first < len(windows):
            indices = list(range(first, min(first + size, len(windows))))
            try:
                outs += asr(
                    [window(i) for i in indices],
                    batch_size=len(indices),
                    return_timestamps=timestamps,
                    generate_kwargs=dict(greedy),
                )
                first += len(indices)
                continue
            except torch.cuda.OutOfMemoryError:
                if len(indices) == 1:
                    raise
            # Retried past the except clause. Inside it, the live exception's traceback
            # holds the failed batch's frames and every tensor in them: each half ran out
            # of memory against the same 18 GB, down to a single window.
            gc.collect()
            torch.cuda.empty_cache()
            size = len(indices) // 2
            print(f"[asr] a batch of {len(indices)} windows did not fit in memory; "
                  f"going on {size} at a time", flush=True)
        # A window that decoded into a loop is redone alone, where the guards'
        # temperature fallback works.
        return [
            asr(window(i), return_timestamps=timestamps, generate_kwargs=dict(generate))
            if _loops(out.get("text") or "")
            else out
            for i, out in enumerate(outs)
        ]

    try:
        outs = decode("word" if words else True)
    except IndexError:
        # Word collation inside transformers indexes past its own list on some
        # generations; segment timing comes from another path and survives.
        print("[asr] word timestamps failed to collate; using segment timing", flush=True)
        outs = decode(True)

    units: list[contracts.Word] = []
    for (s, e), out in zip(windows, outs, strict=True):
        offset, window_end = s / _ASR_RATE, e / _ASR_RATE
        for chunk in out.get("chunks", []):
            text = (chunk.get("text") or "").strip()
            if not text:
                continue
            start, end = chunk["timestamp"]
            start = offset + float(start or 0.0)
            end = offset + float(end) if end is not None else window_end
            units.append(contracts.Word(text=text, start=start, end=max(start, end)))
    return units, detected


def _waveform(audio, rate: int = 16_000) -> dict:
    """Audio decoded in memory, mono at `rate`, in the form the diarization pipeline takes.

    Handed a compressed file, the pipeline's own decoder can come up a few samples short
    of the length it computed -- "resulted in 439895 samples instead of the expected
    441000" -- and fail the whole recording. The service converts to WAV first; this
    decodes with the same ffmpeg the speech recogniser reads files with.
    """
    from transformers.pipelines.audio_utils import ffmpeg_read

    samples = ffmpeg_read(contracts.fetch_bytes(audio), rate)
    return {"waveform": torch.from_numpy(samples).unsqueeze(0), "sample_rate": rate}


def _paddle_name(folder: str) -> str:
    """A bundled PaddleOCR model's own name, from the inference config saved beside it.

    PaddleOCR checks the name it is given against the folder's, so the name has to come
    from the folder rather than from a default or from this repo's config.
    """
    import yaml

    with open(os.path.join(folder, "inference.yml"), encoding="utf-8") as f:
        return yaml.safe_load(f)["Global"]["model_name"]


def _fit_pixels(img, max_pixels: int):
    """Downscale to a pixel budget, preserving aspect ratio.

    Visual tokens are roughly pixels/1024 for this model family, so an unbounded image
    is an unbounded prefill. Upstream the document reader ships scale_to_fit for the same reason;
    this is the dependency-free equivalent.
    """
    pixels = img.width * img.height
    if pixels <= max_pixels:
        return img
    scale = (max_pixels / pixels) ** 0.5
    # Land on the 32px patch grid, or the processor pads and the coordinate frame that
    # data-bbox refers to stops matching the image we measured.
    width = max(32, int(img.width * scale) // 32 * 32)
    height = max(32, int(img.height * scale) // 32 * 32)
    return img.resize((width, height))


def _upscale(img, factor: int):
    """`factor` times the size, on the same 32px grid `_fit_pixels` lands on."""
    from PIL import Image

    width = max(32, img.width * factor // 32 * 32)
    height = max(32, img.height * factor // 32 * 32)
    return img.resize((width, height), Image.LANCZOS)


def _token_budget(img, ceiling: int) -> int:
    """Output budget proportional to how much page there is to describe.

    A stop sign needs a few hundred tokens; a dense page needs thousands. Handing both
    the ceiling means the small case pays the large case's decode latency whenever the
    model declines to emit a stop token.
    """
    approx_visual_tokens = (img.width * img.height) // 1024
    return int(max(512, min(ceiling, approx_visual_tokens * 4)))


def _pixmap_to_image(pixmap):
    from PIL import Image

    return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)


def _ocr_docx(source) -> dict:
    """Read a .docx directly. No rasterising, no OCR.

    The text is already exact in the file; rendering it to pixels and recognising it
    back can only introduce errors. There are no bounding boxes for the same reason --
    a Word document has no fixed pixel layout until it is rendered, so any box would be
    invented. Callers relying on `sections[].lines[].bounds` get empty geometry rather
    than plausible fiction.
    """
    import io as _io
    import xml.etree.ElementTree as ET
    import zipfile

    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with zipfile.ZipFile(_io.BytesIO(contracts.fetch_bytes(source))) as zf:
        root = ET.fromstring(zf.read("word/document.xml"))

    paragraphs = []
    for para in root.iter(f"{{{ns['w']}}}p"):
        text = "".join(node.text or "" for node in para.iter(f"{{{ns['w']}}}t"))
        if text.strip():
            paragraphs.append(text.strip())

    body = "\n".join(paragraphs)
    return {
        "context": body,
        "sections": [{"text": p, "lines": []} for p in paragraphs],
        "has_text": bool(body.strip()),
        "width": 0,
        "height": 0,
        "total_pages": 1,
        "note": "docx text read directly; no bounding boxes exist before rendering",
    }


def _read_text_document(source) -> dict:
    """Return a plain-text file as-is. Nothing to recognise.

    A CSV or log is already exact characters. Rasterising it and reading the pixels
    back could only introduce errors, and previously it was not even attempted -- the
    bytes went to the image reader, which failed with "cannot identify image file" and
    surfaced as a 500. Geometry is empty for the same reason as .docx: a text file has
    no pixel layout until something renders it, so any box would be invented.
    """
    raw = contracts.fetch_bytes(source)
    body = raw.decode("utf-8", errors="replace")
    lines = [line for line in body.splitlines() if line.strip()]
    return {
        "text": body,
        "context": body,
        "sections": [{"text": body, "lines": []}] if body.strip() else [],
        "has_text": bool(body.strip()),
        "width": 0,
        "height": 0,
        "total_pages": 1,
        "line_count": len(lines),
        "note": "text read directly; no bounding boxes exist before rendering",
    }


def _to_numpy(img):
    import numpy as np

    return np.array(img)


# Matching the published contract: outline of the largest region, at most this many
# vertices, simplified until it fits. A shape that cannot reach the cap without a
# tolerance that stops resembling it yields nothing rather than a coarse lie.
_OUTLINE_MAX_POINTS = 40
_OUTLINE_EPSILON = 2.0
_OUTLINE_UNAVAILABLE = False


def _outline_of(mask) -> list[list[int]] | None:
    """Outline of a boolean mask's largest region as [[x, y], ...], or None.

    Only the largest region, so an object split by an occluder is described by its
    main body rather than by a ring around both halves. `bounds` remains the thing to
    read for full extent; this is for shape.
    """
    try:
        import cv2
        import numpy as np
    except Exception as exc:  # pragma: no cover - depends on the image
        # Said once and plainly. Swallowing this returned every object without an
        # outline and no reason anywhere, which reads as "this image has no shapes".
        global _OUTLINE_UNAVAILABLE
        if not _OUTLINE_UNAVAILABLE:
            _OUTLINE_UNAVAILABLE = True
            print(f"[detect] outlines unavailable: {type(exc).__name__}: {exc}", flush=True)
        return None
    try:
        found = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # OpenCV 3 returned (image, contours, hierarchy); 4 returns (contours, hierarchy).
        contours = found[-2]
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        if len(largest) < 3:
            return None

        # Past a tenth of the region's diagonal, simplification no longer resembles the
        # shape, so give up rather than return a triangle standing in for an outline.
        x, y, w, h = cv2.boundingRect(largest)
        ceiling = 0.1 * float(np.hypot(w, h))
        epsilon = _OUTLINE_EPSILON
        simplified = cv2.approxPolyDP(largest, epsilon, True)
        for _ in range(24):
            if len(simplified) <= _OUTLINE_MAX_POINTS:
                break
            epsilon *= 1.5
            if epsilon > ceiling:
                return None
            simplified = cv2.approxPolyDP(largest, epsilon, True)
        if len(simplified) < 3 or len(simplified) > _OUTLINE_MAX_POINTS:
            return None
        return [[int(pt[0][0]), int(pt[0][1])] for pt in simplified]
    except Exception as exc:
        # An outline is an extra; never fail a detection over one -- but say so, or a
        # tracing bug looks exactly like an image with nothing in it.
        print(f"[detect] outline trace failed: {type(exc).__name__}: {exc}", flush=True)
        return None


def _encode_mask_array(mask) -> str:
    """Boolean numpy mask -> base64 PNG, L-mode. Matches the deployed service."""
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray((mask * 255).astype("uint8"), mode="L").save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def _parse_layout(raw: str, img) -> list[contracts.VLMRegion]:
    import json
    import re

    match = re.search(r"\[.*\]", raw, re.S)
    if not match:
        return []
    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []

    regions: list[contracts.VLMRegion] = []
    for item in items:
        bbox = item.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        try:
            bounds = contracts.to_bounds(bbox, src="xyxy", img_w=img.width, img_h=img.height)
        except ValueError:
            continue  # a malformed box must not take the whole page down
        regions.append(
            contracts.VLMRegion(
                text=item.get("text", ""),
                bounds=bounds,
                category=item.get("category", ""),
            )
        )
    return regions

