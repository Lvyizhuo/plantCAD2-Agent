"""PlantCAD2 Inference Engine.

Central orchestrator that manages model loading and dispatches inference requests
to the appropriate modules.
"""

import logging
from pathlib import Path
from typing import Optional

import torch
from peft import PeftModel

from modules.model_loader import load_mlm_model, load_cls_model, load_lora_adapter
from modules.embedding import extract_embeddings
from modules.variant_score import score_variant as _score_variant
from modules.lora_predict import predict_with_lora
from modules.masked_predict import masked_predict as _masked_predict

logger = logging.getLogger(__name__)

# Task name -> (LoRA directory name, task_type, num_labels, problem_type)
TASK_REGISTRY = {
    "acr_arabidopsis": (
        "cross_species_acr_train_on_arabidopsis_plantcad2_large",
        "classification", 2, None,
    ),
    "acr_nine_species": (
        "cross_species_acr_train_on_nine_species_plantcad2_large",
        "classification", 2, None,
    ),
    "acr_cell_type": (
        "cell_type_specific_acr_plantcad2_large",
        "classification", 2, None,
    ),
    "expression_on_off": (
        "cross_species_leaf_on_off_expression_plantcad2_large",
        "classification", 2, None,
    ),
    "expression_absolute": (
        "cross_species_leaf_absolute_expression_plantcad2_large",
        "regression", 1, "regression",
    ),
    "translation_on_off": (
        "cross_species_leaf_on_off_translation_plantcad2_large",
        "classification", 2, None,
    ),
    "translation_absolute": (
        "cross_species_leaf_absolute_translation_plantcad2_large",
        "regression", 1, "regression",
    ),
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

        # Load CLS model (for LoRA tasks)
        logger.info("Loading CLS model...")
        self.cls_model = load_cls_model(base_model_path, device)

        # LoRA adapter cache: task_name -> PeftModel
        self._lora_cache: dict[str, PeftModel] = {}

        if preload_lora:
            self._preload_all_lora()

    def _preload_all_lora(self):
        """Preload all registered LoRA adapters."""
        for task_name, (lora_dir, task_type, num_labels, problem_type) in TASK_REGISTRY.items():
            lora_path = self.lora_models_path / lora_dir
            if lora_path.exists():
                try:
                    # For regression tasks, we need a different base model config
                    # But the CLS model is already loaded with default config (num_labels=2)
                    # We'll load adapters on-the-fly with correct config for regression tasks
                    if task_type == "regression":
                        logger.info(
                            f"Skipping preload of '{task_name}' (regression) — "
                            f"will load on-demand with correct config"
                        )
                        continue
                    adapter = load_lora_adapter(
                        self.cls_model,
                        str(lora_path),
                        base_model_path=self.base_model_path,
                    )
                    self._lora_cache[task_name] = adapter
                    logger.info(f"Preloaded LoRA adapter: {task_name}")
                except Exception as e:
                    logger.warning(f"Failed to preload LoRA '{task_name}': {e}")
            else:
                logger.warning(f"LoRA directory not found: {lora_path}")

    def _get_lora_model(self, task_name: str) -> PeftModel:
        """Get a LoRA model for the given task, loading on-demand if needed."""
        if task_name in self._lora_cache:
            return self._lora_cache[task_name]

        if task_name not in TASK_REGISTRY:
            raise ValueError(
                f"Unknown task '{task_name}'. "
                f"Available: {list(TASK_REGISTRY.keys())}"
            )

        lora_dir, task_type, num_labels, problem_type = TASK_REGISTRY[task_name]
        lora_path = self.lora_models_path / lora_dir
        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA directory not found: {lora_path}")

        # For regression tasks, load a fresh CLS model with correct config
        if task_type == "regression":
            logger.info(f"Loading regression CLS model for '{task_name}'")
            base_model = load_cls_model(
                self.base_model_path,
                self.device,
                num_labels=1,
                problem_type="regression",
            )
        else:
            base_model = self.cls_model

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
        _, task_type, _, _ = TASK_REGISTRY[task]
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
