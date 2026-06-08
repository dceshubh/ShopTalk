"""Caption-model loading and inference — the comparison harness behind the "compare >= 2
caption models" requirement (BLIP-2 vs BLIP-base, per `models.captioning` in config.yaml).

Both families are loaded through their explicit transformers classes, not `AutoModel` —
BLIP-2 (a Q-Former + frozen LLM architecture) and BLIP-base (a single encoder-decoder) use
different processors and generation signatures, so an Auto* class would paper over an
architecture difference that's actually part of the comparison story, not hide it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image
from transformers import (
    Blip2ForConditionalGeneration,
    Blip2Processor,
    BlipForConditionalGeneration,
    BlipProcessor,
)

from src.common.device import resolve_device
from src.common.logging import get_logger

logger = get_logger(__name__)

# model name -> (processor class, model class). Add an entry here before captioning with
# a new model — the explicit registry is what keeps `load_captioner` from guessing.
_MODEL_REGISTRY: dict[str, tuple[type, type]] = {
    "Salesforce/blip2-opt-2.7b": (Blip2Processor, Blip2ForConditionalGeneration),
    "Salesforce/blip-image-captioning-base": (BlipProcessor, BlipForConditionalGeneration),
}


@dataclass
class Captioner:
    """A loaded caption model + its processor, pinned to one device and dtype."""

    model_name: str
    processor: object
    model: object
    device: str

    def caption(self, image: Image.Image, *, max_new_tokens: int = 30) -> str:
        """Generate one unconditional ("describe this image") caption for `image`.

        BLIP-2 needs `text=""` so its processor prepends `num_query_tokens` `<image>`
        placeholder tokens to `input_ids` (set on the processor in `load_captioner`) —
        without them, `Blip2ForConditionalGeneration.generate` has nowhere to splice the
        Q-Former's image embeddings and raises a shape-mismatch error. BLIP-base has no
        such attribute and runs unconditional captioning with `text=None` as before.
        """
        text = "" if hasattr(self.processor, "num_query_tokens") else None
        inputs = self.processor(images=image, text=text, return_tensors="pt")
        inputs = {
            k: (v.to(self.device, self.model.dtype) if v.is_floating_point() else v.to(self.device))
            for k, v in inputs.items()
        }
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text.strip()


def load_captioner(model_name: str, *, device: str | None = None) -> Captioner:
    """Load a registered caption model onto `device` (auto-detected if not given).

    Uses fp16 on GPU/MPS — material for `blip2-opt-2.7b`'s ~5.4 GB fp32 footprint on a
    16 GB M3 — and fp32 on CPU, where fp16 matmul kernels aren't reliably available.
    """
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown caption model {model_name!r} — register its (processor, model) "
            "classes in _MODEL_REGISTRY before loading it."
        )
    processor_cls, model_cls = _MODEL_REGISTRY[model_name]
    device = device or resolve_device()
    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

    logger.info("Loading caption model %s onto %s (dtype=%s)", model_name, device, dtype)
    processor = processor_cls.from_pretrained(model_name)
    model = model_cls.from_pretrained(model_name, torch_dtype=dtype).to(device)
    model.eval()
    if hasattr(processor, "num_query_tokens"):
        # Cached processor configs predate this attribute; without it the processor never
        # expands `<image>` placeholder tokens and `generate` fails (see `Captioner.caption`).
        processor.num_query_tokens = model.config.num_query_tokens
    return Captioner(model_name=model_name, processor=processor, model=model, device=device)
