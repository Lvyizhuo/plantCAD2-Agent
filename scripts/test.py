# Test core dependencies
import torch
from mamba_ssm import Mamba
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Test PlantCAD model loading
tokenizer = AutoTokenizer.from_pretrained('./../models/PlantCAD2-Large-l48-d1536')
model = AutoModelForMaskedLM.from_pretrained('./../models/PlantCAD2-Large-l48-d1536', trust_remote_code=True)
device = 'cuda:0'
model.to(device)
print("✅ Installation successful!")