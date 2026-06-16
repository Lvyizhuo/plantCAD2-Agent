"""Service configuration for PlantCAD2 inference API."""

import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    base_model_path: str = os.getenv(
        "PLANTCAD2_MODEL_PATH", "models/PlantCAD2-Large-l48-d1536"
    )
    lora_models_path: str = os.getenv("PLANTCAD2_LORA_PATH", "models")
    device: str = os.getenv("PLANTCAD2_DEVICE", "cuda:0")
    preload_lora: bool = os.getenv("PLANTCAD2_PRELOAD_LORA", "true").lower() == "true"

    # Sequence length limits
    max_sequence_length: int = 8192

    # Task -> LoRA directory mapping (mirrors engine.TASK_REGISTRY)
    task_lora_map: dict = field(default_factory=lambda: {
        "acr_arabidopsis": "cross_species_acr_train_on_arabidopsis_plantcad2_large",
        "acr_nine_species": "cross_species_acr_train_on_nine_species_plantcad2_large",
        "acr_cell_type": "cell_type_specific_acr_plantcad2_large",
        "expression_on_off": "cross_species_leaf_on_off_expression_plantcad2_large",
        "expression_absolute": "cross_species_leaf_absolute_expression_plantcad2_large",
        "translation_on_off": "cross_species_leaf_on_off_translation_plantcad2_large",
        "translation_absolute": "cross_species_leaf_absolute_translation_plantcad2_large",
    })


settings = Settings()
