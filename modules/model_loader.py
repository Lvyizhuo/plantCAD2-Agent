"""Model loading utilities for PlantCAD2.

Extracted and adapted from src/zero_shot_score.py and src/lora_fine_tune.py.
Supports local/offline model loading without HuggingFace remote access.
"""

import inspect
import logging
from pathlib import Path
from typing import Optional

import torch
from safetensors import safe_open
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from peft import PeftModel

logger = logging.getLogger(__name__)


def get_optimal_dtype() -> torch.dtype:
    """Select the best dtype for the current GPU."""
    if not torch.cuda.is_available():
        logger.info("No GPU available, using float32")
        return torch.float32

    device_index = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device_index)

    if capability[0] >= 8:
        logger.info("GPU supports sm_80+, using bfloat16")
        return torch.bfloat16
    elif capability[0] >= 6:
        logger.info("GPU supports sm_60+, using float16")
        return torch.float16
    else:
        logger.info("GPU does not support float16/bfloat16, using float32")
        return torch.float32


def load_mlm_model(
    model_path: str,
    device: str = "cuda:0",
    dtype: Optional[torch.dtype] = None,
) -> tuple:
    """Load the Masked LM model (AutoModelForMaskedLM) and tokenizer.

    Used for: embedding extraction, masked token prediction, variant scoring.

    Args:
        model_path: Local path to the base model directory.
        device: Target device.
        dtype: Model dtype. If None, auto-detect optimal dtype.

    Returns:
        (model, tokenizer) tuple.
    """
    if dtype is None:
        dtype = get_optimal_dtype()

    logger.info(f"Loading MLM model from {model_path} with dtype={dtype}")

    try:
        model = AutoModelForMaskedLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
        model.to(dtype)
    except Exception as e:
        logger.warning(f"Failed to load with {dtype}, falling back to float32: {e}")
        model = AutoModelForMaskedLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model.to(device)
    model.eval()
    logger.info("MLM model loaded successfully")
    return model, tokenizer


def load_cls_model(
    model_path: str,
    device: str = "cuda:0",
    dtype: Optional[torch.dtype] = None,
    num_labels: int = 2,
    problem_type: Optional[str] = None,
) -> AutoModelForSequenceClassification:
    """Load the Sequence Classification model (AutoModelForSequenceClassification).

    Used for: LoRA fine-tuned task predictions.

    Args:
        model_path: Local path to the base model directory.
        device: Target device.
        dtype: Model dtype. If None, auto-detect optimal dtype.
        num_labels: Number of output labels (default 2 for binary classification).
        problem_type: "regression", "multi_label_classification", or None.

    Returns:
        The classification model.
    """
    if dtype is None:
        dtype = get_optimal_dtype()

    logger.info(f"Loading CLS model from {model_path} with dtype={dtype}")

    kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
    }
    if num_labels is not None:
        kwargs["num_labels"] = num_labels
    if problem_type is not None:
        kwargs["problem_type"] = problem_type

    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path, **kwargs)
        model.to(dtype)
    except Exception as e:
        logger.warning(f"Failed to load with {dtype}, falling back to float32: {e}")
        kwargs["torch_dtype"] = torch.float32
        model = AutoModelForSequenceClassification.from_pretrained(model_path, **kwargs)

    model.to(device)
    model.eval()

    # Caduceus forward() does not accept standard HF kwargs like attention_mask
    # or output_attentions, but PeftModel injects them automatically.
    # Wrap forward to strip all unsupported kwargs.
    _original_forward = model.forward
    _supported_params = set(inspect.signature(_original_forward).parameters.keys())

    def _forward_filtered(*args, **kwargs):
        filtered = {k: v for k, v in kwargs.items() if k in _supported_params}
        return _original_forward(*args, **filtered)

    model.forward = _forward_filtered

    logger.info("CLS model loaded successfully")
    return model


def detect_adapter_num_labels(lora_path: str) -> int:
    """Detect num_labels from adapter weights by reading score.weight shape."""
    adapter_file = Path(lora_path) / "adapter_model.safetensors"
    if not adapter_file.exists():
        return 2  # default binary classification
    with safe_open(str(adapter_file), framework="pt") as f:
        for key in f.keys():
            if key.endswith("score.weight"):
                return f.get_tensor(key).shape[0]
    return 2


def load_lora_adapter(
    base_model: AutoModelForSequenceClassification,
    lora_path: str,
    base_model_path: Optional[str] = None,
) -> PeftModel:
    """Load a LoRA adapter onto a base classification model.

    The adapter_config.json may reference a remote HuggingFace model ID.
    We override it with the local base_model_path for offline loading.

    Args:
        base_model: The base AutoModelForSequenceClassification model.
        lora_path: Local path to the LoRA adapter directory.
        base_model_path: Local path to base model (overrides adapter config).

    Returns:
        PeftModel with the LoRA adapter applied.
    """
    logger.info(f"Loading LoRA adapter from {lora_path}")

    # If local base model path provided, temporarily override the adapter config
    # to avoid trying to download from HuggingFace
    if base_model_path is not None:
        from peft import PeftConfig
        config = PeftConfig.from_pretrained(lora_path)
        original_path = config.base_model_name_or_path
        if original_path != base_model_path:
            logger.info(
                f"Overriding adapter base_model from '{original_path}' to '{base_model_path}'"
            )
            # PeftModel.from_pretrained loads the config internally, so we need to
            # pass the base model directly (which we already have)
            pass

    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()
    logger.info(f"LoRA adapter loaded from {lora_path}")
    return model
