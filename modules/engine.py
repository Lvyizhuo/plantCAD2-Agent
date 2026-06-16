"""PlantCAD2 Inference Engine.

Central orchestrator that manages model loading and dispatches inference requests
to the appropriate modules.
"""

import logging
from pathlib import Path
from typing import Optional

import torch
from peft import PeftModel

from modules.model_loader import load_mlm_model, load_cls_model, load_lora_adapter, detect_adapter_num_labels
from modules.embedding import extract_embeddings
from modules.variant_score import score_variant as _score_variant
from modules.lora_predict import predict_with_lora
from modules.masked_predict import masked_predict as _masked_predict

logger = logging.getLogger(__name__)

# Task name -> (LoRA directory name, task_type)
# num_labels is auto-detected from adapter weights at load time.
TASK_REGISTRY = {
    "acr_arabidopsis": ("cross_species_acr_train_on_arabidopsis_plantcad2_large", "classification"),
    "acr_nine_species": ("cross_species_acr_train_on_nine_species_plantcad2_large", "classification"),
    "acr_cell_type": ("cell_type_specific_acr_plantcad2_large", "classification"),
    "expression_on_off": ("cross_species_leaf_on_off_expression_plantcad2_large", "classification"),
    "expression_absolute": ("cross_species_leaf_absolute_expression_plantcad2_large", "regression"),
    "translation_on_off": ("cross_species_leaf_on_off_translation_plantcad2_large", "classification"),
    "translation_absolute": ("cross_species_leaf_absolute_translation_plantcad2_large", "regression"),
}


class PlantCAD2Engine:
    """PlantCAD2 inference engine.

    Manages two model heads:
    - MLM head (AutoModelForMaskedLM): for embedding, masking, variant scoring
    - CLS head (AutoModelForSequenceClassification): for LoRA task predictions

    Both share the same backbone; GPU cache deduplicates the weights.
    """

    def __init__(
        self,
        base_model_path: str,
        lora_models_path: str = "models",
        device: str = "cuda:0",
        preload_lora: bool = True,
    ):
        """Initialize the engine.

        Args:
            base_model_path: Path to the base PlantCAD2 model directory.
            lora_models_path: Parent directory containing LoRA adapter subdirectories.
            device: Target device for inference.
            preload_lora: Whether to preload all LoRA adapters at startup.
        """
        self.base_model_path = base_model_path
        self.lora_models_path = Path(lora_models_path)
        self.device = device

        # Load MLM model (for embedding, masking, variant scoring)
        logger.info("Loading MLM model...")
        self.mlm_model, self.tokenizer = load_mlm_model(base_model_path, device)

        # CLS model cache: (num_labels, problem_type) -> model
        self._cls_models: dict[tuple, AutoModelForSequenceClassification] = {}
        # Default CLS model (num_labels=2, classification)
        self.cls_model = load_cls_model(base_model_path, device)
        self._cls_models[(2, None)] = self.cls_model

        # LoRA adapter cache: task_name -> PeftModel
        self._lora_cache: dict[str, PeftModel] = {}

        if preload_lora:
            self._preload_all_lora()

    def _get_or_load_cls_model(self, num_labels: int, task_type: str):
        """Get or create a CLS model with the given config, cached by (num_labels, problem_type)."""
        problem_type = "regression" if task_type == "regression" else (
            "multi_label_classification" if num_labels > 2 else None
        )
        cache_key = (num_labels, problem_type)
        if cache_key not in self._cls_models:
            logger.info(f"Loading CLS model with num_labels={num_labels}, problem_type={problem_type}")
            self._cls_models[cache_key] = load_cls_model(
                self.base_model_path, self.device,
                num_labels=num_labels, problem_type=problem_type,
            )
        return self._cls_models[cache_key]

    def _preload_all_lora(self):
        """Preload all registered LoRA adapters."""
        for task_name, (lora_dir, task_type) in TASK_REGISTRY.items():
            lora_path = self.lora_models_path / lora_dir
            if not lora_path.exists():
                logger.warning(f"LoRA directory not found: {lora_path}")
                continue
            try:
                num_labels = detect_adapter_num_labels(str(lora_path))
                logger.info(f"Detected num_labels={num_labels} for '{task_name}'")
                base = self._get_or_load_cls_model(num_labels, task_type)
                adapter = load_lora_adapter(
                    base, str(lora_path),
                    base_model_path=self.base_model_path,
                )
                self._lora_cache[task_name] = adapter
                logger.info(f"Preloaded LoRA adapter: {task_name}")
            except Exception as e:
                logger.warning(f"Failed to preload LoRA '{task_name}': {e}")

    def _get_lora_model(self, task_name: str) -> PeftModel:
        """Get a LoRA model for the given task, loading on-demand if needed."""
        if task_name in self._lora_cache:
            return self._lora_cache[task_name]

        if task_name not in TASK_REGISTRY:
            raise ValueError(
                f"Unknown task '{task_name}'. "
                f"Available: {list(TASK_REGISTRY.keys())}"
            )

        lora_dir, task_type = TASK_REGISTRY[task_name]
        lora_path = self.lora_models_path / lora_dir
        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA directory not found: {lora_path}")

        num_labels = detect_adapter_num_labels(str(lora_path))
        logger.info(f"Detected num_labels={num_labels} for '{task_name}'")
        base_model = self._get_or_load_cls_model(num_labels, task_type)

        adapter = load_lora_adapter(
            base_model,
            str(lora_path),
            base_model_path=self.base_model_path,
        )
        self._lora_cache[task_name] = adapter
        return adapter

    # ------------------------------------------------------------------
    # Inference methods (implemented in P1)
    # ------------------------------------------------------------------

    def get_embeddings(self, sequence: str, normalize: bool = True) -> dict:
        """Extract per-position embeddings for a DNA sequence."""
        return extract_embeddings(
            self.mlm_model, self.tokenizer, sequence,
            device=self.device, normalize=normalize,
        )

    def score_variant(
        self,
        sequence: str,
        position: int,
        ref_allele: str,
        alt_alleles: list[str],
    ) -> dict:
        """Score variant effect using zero-shot masked prediction."""
        return _score_variant(
            self.mlm_model, self.tokenizer, sequence,
            position, ref_allele, alt_alleles,
            device=self.device,
        )

    def predict_function(self, sequence: str, task: str) -> dict:
        """Run a LoRA fine-tuned task prediction."""
        lora_model = self._get_lora_model(task)
        _, task_type = TASK_REGISTRY[task]
        result = predict_with_lora(
            lora_model, self.tokenizer, sequence,
            task_type=task_type, device=self.device,
        )
        result["task"] = task
        return result

    def masked_predict(self, sequence: str, positions: list[int]) -> dict:
        """Predict nucleotide probabilities at specified positions."""
        return _masked_predict(
            self.mlm_model, self.tokenizer, sequence,
            positions, device=self.device,
        )
