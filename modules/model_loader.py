"""Model loading utilities for PlantCAD2.

Extracted and adapted from src/zero_shot_score.py and src/lora_fine_tune.py.
Supports local/offline model loading without HuggingFace remote access.
"""

import inspect
from pathlib import Path
from typing import Optional, Tuple

import torch
from loguru import logger
from safetensors.torch import load_file
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from peft import PeftModel


def get_optimal_dtype() -> torch.dtype:
    """Select the best dtype for the current GPU.

    Returns:
        torch.bfloat16 for sm_80+ (A100, H100)
        torch.float16 for sm_60+ (V100, T4)
        torch.float32 for CPU or older GPUs
    """
    if not torch.cuda.is_available():
        logger.info("No GPU available, using float32")
        return torch.float32

    device_index = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device_index)
    sm_version = capability[0] * 10 + capability[1]

    if sm_version >= 80:
        logger.info(f"GPU sm_{sm_version} detected, using bfloat16")
        return torch.bfloat16
    elif sm_version >= 60:
        logger.info(f"GPU sm_{sm_version} detected, using float16")
        return torch.float16
    else:
        logger.info(f"GPU sm_{sm_version} detected, using float32")
        return torch.float32


def load_mlm_model(
    model_path: str,
    device: str = "cuda:0",
    dtype: Optional[torch.dtype] = None,
) -> Tuple[AutoModelForMaskedLM, AutoTokenizer]:
    """Load the Masked LM model and tokenizer.

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
    """Load the Sequence Classification model.

    Used for: LoRA fine-tuned task predictions.

    Args:
        model_path: Local path to the base model directory.
        device: Target device.
        dtype: Model dtype. If None, defaults to float16 (mamba-ssm CUDA kernels
            do not support bfloat16, and LoRA adapter weights are stored in float32).
        num_labels: Number of output labels (default 2 for binary classification).
        problem_type: "regression", "multi_label_classification", or None.

    Returns:
        The classification model with wrapped forward method.
    """
    if dtype is None:
        # mamba-ssm CUDA kernels do not support BFloat16, so we use float16
        # instead of get_optimal_dtype() which returns bfloat16 on A100.
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

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
    """Detect num_labels from adapter weights by reading score.weight shape.

    Args:
        lora_path: Path to the LoRA adapter directory.

    Returns:
        Number of labels (e.g., 2 for binary, 92 for cell_type, 1 for regression).
    """
    adapter_file = Path(lora_path) / "adapter_model.safetensors"
    if not adapter_file.exists():
        logger.warning(f"Adapter file not found: {adapter_file}, defaulting to num_labels=2")
        return 2

    state_dict = load_file(str(adapter_file))
    for key in state_dict:
        if key.endswith("score.weight"):
            num_labels = state_dict[key].shape[0]
            logger.info(f"Detected num_labels={num_labels} from adapter")
            return num_labels

    logger.warning("Could not detect num_labels from adapter, defaulting to 2")
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

    if base_model_path is not None:
        # Override the base_model_name_or_path to point to local files
        import json
        config_path = Path(lora_path) / "adapter_config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)
            original_path = config.get("base_model_name_or_path", "")
            config["base_model_name_or_path"] = base_model_path
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            logger.debug(f"Overrode base_model_path: {original_path} -> {base_model_path}")

    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()
    logger.info("LoRA adapter loaded successfully")
    return model
