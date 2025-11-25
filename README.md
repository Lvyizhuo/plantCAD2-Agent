![Static Badge](https://img.shields.io/badge/Linux-blue?logo=Linux&logoColor=white)
![GitHub Repo stars](https://img.shields.io/github/stars/kuleshov-group/PlantCaduceus)
![GitHub Issues or Pull Requests](https://img.shields.io/github/issues/kuleshov-group/PlantCaduceus)
[![PlantCAD](https://zenodo.org/badge/DOI/10.1073/pnas.2421738122.svg)](https://doi.org/10.1073/pnas.2421738122)
[![PlantCAD2](https://zenodo.org/badge/DOI/10.1101/2025.08.27.672609.svg)](https://doi.org/10.1101/2025.08.27.672609)
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-yellow.svg?style=flat)](https://huggingface.co/collections/kuleshov-group/plantcaduceus-512bp-len-665a229ee098db706a55e44a)
<a href="https://huggingface.co/collections/kuleshov-group/plantcaduceus-512bp-len-665a229ee098db706a55e44a">
  <img alt="Hugging Face Downloads" src="https://img.shields.io/badge/dynamic/json?color=blue&label=downloads&query=downloads&url=https://huggingface.co/api/models/kuleshov-group/PlantCaduceus_l32">
</a>
<p align="center">
  <img src="img/logo.jpg" alt="logo" width="20%">
</p>

## [🚀 PlantCAD2 Release!](https://www.biorxiv.org/content/10.1101/2025.08.27.672609v1)
We’re excited to announce [PlantCAD2](https://huggingface.co/collections/kuleshov-group/plantcad2-67e437e241a382671371a572) 🌱 — our new DNA foundation model for angiosperms.

In addition, we’re also releasing a collection of [LoRA fine-tuned models](https://huggingface.co/collections/plantcad/fine-tuned-plantcad2-models-68b316a57616134fa7a1b6b6) 🎯, tailored for key downstream tasks including accessible chromatin, gene expression, and protein translation.

- Explore the **fine-tuned** PlantCAD2 models [here](docs/PlantCAD2-overview.md)

- Explore the **zero-shot** evaluation of PlantCAD2 models [here](docs/zero-shot-eval.md)

- Explore the post-training or pre-training of PlantCAD2 [here](https://github.com/kuleshov-group/PlantCaduceus/issues/19)

## Table of Contents
- [PlantCAD overview](#plantcad-overview)
- [Quick Start](#quick-start)
- [Model summary](#model-summary)
- [Prerequisites and system requirements](#prerequisites-and-system-requirements)
- [Installation](#installation)
  - [Option 1: Google Colab (Recommended for beginners)](#option-1-google-colab-recommended-for-beginners)
  - [Option 2: Local installation](docs/local-install.md)
  - [Option 3: Using Docker](docker/README.md)
- [Basic Usage](#basic-usage)
  - [Exploring model inputs and outputs](#exploring-model-inputs-and-outputs)
  - [Zero-shot Scoring of Genomic Variants and Regions](#zero-shot-scoring-of-genomic-variants-and-regions)
  - [In-silico mutagenesis pipeline](#in-silico-mutagenesis-pipeline)
- [Advanced Usage](#advanced-usage)
  - [Training XGBoost classifiers](#training-xgboost-classifiers)
  - [Using pre-trained XGBoost classifiers](#using-pre-trained-xgboost-classifiers)
- [Development and Training](#development-and-training)
  - [Pre-training PlantCAD](#pre-training-plantcad)
  - [Performance benchmarks](#performance-benchmarks)
- [Citation](#citation)

## [PlantCAD overview](https://plantcaduceus.github.io/)

PlantCaduceus, with its short name of **PlantCAD**, is a plant DNA LM based on the [Caduceus](https://arxiv.org/abs/2403.03234) architecture, which extends the efficient [Mamba](https://arxiv.org/abs/2312.00752) linear-time sequence modeling framework to incorporate bi-directionality and reverse complement equivariance, specifically designed for DNA sequences. PlantCAD is pre-trained on a curated dataset of 16 Angiosperm genomes. PlantCAD showed state-of-the-art cross species performance in predicting TIS, TTS, Splice Donor and Splice Acceptor. The zero-shot of PlantCAD enables identifying genome-wide deleterious mutations and known causal variants in Arabidopsis, Sorghum and Maize.

## Quick Start

**New to PlantCAD?** Try our [Google Colab demo](https://colab.research.google.com/drive/1QW9Lgwra0vHQAOICE2hsIVcp6DKClyhO?usp=sharing) - no installation required!

**For local usage:** See installation instructions [here](docs/local-install.md), then use `notebooks/examples.ipynb` to get started.

## Model summary
Pre-trained PlantCAD models have been uploaded to [HuggingFace 🤗](https://huggingface.co/collections/kuleshov-group/plantcaduceus-512bp-len-665a229ee098db706a55e44a). Here's the summary of four PlantCAD models with different parameter sizes.
| Model | Sequence Length | Model Size | Embedding Size |
|-------|----------------|------------|----------------|
| [PlantCaduceus_l20](https://huggingface.co/kuleshov-group/PlantCaduceus_l20) | 512bp | 20M | 384 |
| [PlantCaduceus_l24](https://huggingface.co/kuleshov-group/PlantCaduceus_l24) | 512bp | 40M | 512 |
| [PlantCaduceus_l28](https://huggingface.co/kuleshov-group/PlantCaduceus_l28) | 512bp | 128M | 768 |
| [PlantCaduceus_l32](https://huggingface.co/kuleshov-group/PlantCaduceus_l32) | 512bp | 225M | 1024 |

**Model Selection Guide:**
- PlantCaduceus_l20: Good for testing and quick analysis
- PlantCaduceus_l32: **Recommended** for research and production (best performance)

## Prerequisites and System Requirements

**For Google Colab:** Just a Google account - GPU runtime recommended (free tier available)

**For Local Installation: NVIDIA GPU is required!**

## Installation

### Option 1: Google Colab (Recommended for beginners)
**No installation required!** Just open our [PlantCAD Google Colab notebook](https://colab.research.google.com/drive/1QW9Lgwra0vHQAOICE2hsIVcp6DKClyhO?usp=sharing) and start analyzing your data.

### Option 2: Local installation
See our [Local Installation Guide](docs/local-install.md).


## Basic Usage

### Exploring model inputs and outputs

The easiest way to start is with our example notebook: `notebooks/examples.ipynb`

**Quick example - Get sequence embeddings:**
```python
import torch
from mamba_ssm import Mamba
from transformers import AutoTokenizer, AutoModelForMaskedLM
import pandas as pd

device = 'cuda:0'
# Test PlantCAD model loading
tokenizer = AutoTokenizer.from_pretrained('kuleshov-group/PlantCaduceus_l32')
model = AutoModelForMaskedLM.from_pretrained('kuleshov-group/PlantCaduceus_l32', trust_remote_code=True)
model.to(device)
# Example plant DNA sequence (512bp max)
sequence = "CTTAATTAATATTGCCTTTGTAATAACGCGCGAAACACAAATCTTCTCTGCCTAATGCAGTAGTCATGTGTTGACTCCTTCAAAATTTCCAAGAAGTTAGTGGCTGGTGTGTCATTGTCTTCATCTTTTTTTTTTTTTTTTTAAAAATTGAATGCGACATGTACTCCTCAACGTATAAGCTCAATGCTTGTTACTGAAACATCTCTTGTCTGATTTTTTCAGGCTAAGTCTTACAGAAAGTGATTGGGCACTTCAATGGCTTTCACAAATGAAAAAGATGGATCTAAGGGATTTGTGAAGAGAGTGGCTTCATCTTTCTCCATGAGGAAGAAGAAGAATGCAACAAGTGAACCCAAGTTGCTTCCAAGATCGAAATCAACAGGTTCTGCTAACTTTGAATCCATGAGGCTACCTGCAACGAAGAAGATTTCAGATGTCACAAACAAAACAAGGATCAAACCATTAGGTGGTGTAGCACCAGCACAACCAAGAAGGGAAAAGATCGATGATCG"
device = 'cuda:0'
# Get embeddings
encoding = tokenizer.encode_plus(
            sequence,
            return_tensors="pt",
            return_attention_mask=False,
            return_token_type_ids=False
        )

input_ids = encoding["input_ids"].to(device)
with torch.inference_mode():
    outputs = model(input_ids=input_ids, output_hidden_states=True)

embeddings = outputs.hidden_states[-1]
print(f"Embedding shape: {embeddings.shape}")  # [batch_size, seq_len, embedding_dim]

embeddings = embeddings.to(torch.float32).cpu().numpy()
# Given that PlantCaduceus has bi-directionality and reverse complement equivariance, so the first half of embedding is for forward sequences and the sencond half is for reverse complemented sequences, we need to average the embeddings before working on downstream classifier

hidden_size = embeddings.shape[-1] // 2
forward = embeddings[..., 0:hidden_size]
reverse = embeddings[..., hidden_size:]
reverse = reverse[..., ::-1]
averaged_embeddings = (forward + reverse) / 2
print(averaged_embeddings.shape)
```

### Zero-shot Scoring of Genomic Variants and Regions

The `zero_shot_score.py` script now provides unified functionality to estimate the functional impact of genetic variants or score genomic regions using PlantCAD's log-likelihood scores. It supports two primary modes:

1.  **Variant Scoring (VCF Input):** Scores specific genetic variants provided in a VCF file.
2.  **Genome-Wide Region Scoring (BED Input):** Calculates log-likelihood ratios for all positions within specified genomic regions (BED file).

```bash
# Download example reference genome
wget https://download.maizegdb.org/Zm-B73-REFERENCE-NAM-5.0/Zm-B73-REFERENCE-NAM-5.0.fa.gz
gunzip Zm-B73-REFERENCE-NAM-5.0.fa.gz
```

```bash
# VCF Input Mode
# --- Example: Variant Scoring (VCF Input) ---
# Estimate impact of specific variants from a VCF file.
# Note: Only the first 8 columns of the VCF file (CHROM, POS, ID, REF, ALT, QUAL, FILTER, INFO) are strictly required.
python src/zero_shot_score.py \
    -input-vcf examples/example_maize_snp.vcf \
    -input-fasta Zm-B73-REFERENCE-NAM-5.0.fa \
    -output scored_variants.vcf \
    -model 'kuleshov-group/PlantCaduceus_l32' \
    -device 'cuda:0'

# Expected output for VCF mode:
# - A new VCF file ('scored_variants.vcf') with PlantCAD scores added to the INFO field.
# - Scores represent log-likelihood ratios between reference and alternative alleles.
#   Low negative scores indicate potentially more deleterious mutations.
```

```bash
# BED Input Mode
# --- Example: Genome-Wide Region Scoring (BED Input) ---
# Calculate log-likelihood ratios for all positions within specified BED regions.
# Note: You would need an example BED file for this.
# For demonstration, creating a dummy BED file:
echo -e "chr1\t1000\t1010\nchr1\t2000\t2015" > examples/example_regions.bed

python src/zero_shot_score.py \
    -input-bed examples/example_regions.bed \
    -input-fasta Zm-B73-REFERENCE-NAM-5.0.fa \
    -output genome_wide_scores.tsv \
    -model 'kuleshov-group/PlantCaduceus_l32' \
    -device 'cuda:0' \
    -step-size 1 \
    -aggregation average \
    -use-masking \
    -output-raw-prob

# Expected output for BED mode:
# - A tab-separated file ('genome_wide_scores.tsv') containing scores for each position.
# - Output includes chromosome, start, end, reference allele, aggregated score,
#   and optionally raw probabilities for all four nucleotides.
# - `-step-size`: Number of positions to analyze per window, if step size is greater than 1, we recommend turn off masking!!!
# - `-aggregation`: How to aggregate alternative allele scores.
#   - `'max'`: Reports the maximum log-likelihood ratio among all three alternative alleles relative to the reference.
#   - `'average'`: Reports the average log-likelihood ratio across all three alternative alleles relative to the reference.
#   - `'all'`: Reports the individual log-likelihood ratios for each of the three alternative alleles relative to the reference.
# - `-use-masking`: Whether to mask the central position(s) during inference.
# - `-output-raw-prob`: Include raw probabilities in the output.
```

When analyzing the entire genome or large genomic regions, the `-step-size` parameter is very important for speeding up the analysis. For a detailed guide on this trade-off between speed and accuracy, see **[here](docs/step_size_genome_wide_llr.md)**.


### In-silico mutagenesis pipeline

For large-scale simulation and analysis of genetic variants, we provide a comprehensive in-silico mutagenesis pipeline. See [pipelines/in-silico-mutagenesis/README.md](pipelines/in-silico-mutagenesis/README.md) for detailed instructions.


## Advanced Usage

### Training XGBoost classifiers

Train custom classifiers on top of PlantCAD embeddings for specific annotation tasks (e.g., TIS, TTS, splice sites).

**Purpose**: Fine-tune prediction performance for specific annotation tasks using supervised learning.

**Data format**: Training data should follow the format used in our [cross-species annotation dataset](https://huggingface.co/datasets/kuleshov-group/cross-species-single-nucleotide-annotation/tree/main/TIS).

```bash
python src/train_XGBoost.py \
    -train train.tsv \
    -valid valid.tsv \
    -test test_rice.tsv \
    -model 'kuleshov-group/PlantCaduceus_l20' \
    -output ./output \
    -device 'cuda:0'
```

**Expected outputs:**
- Trained XGBoost classifier (`.json` file)
- Performance metrics on validation/test sets
- Feature importance analysis

### Using pre-trained XGBoost classifiers

We provide pre-trained XGBoost classifiers for common annotation tasks in the [`classifiers`](classifiers) directory.

**Available classifiers:**
- TIS (Translation Initiation Sites)
- TTS (Translation Termination Sites)  
- Splice donor/acceptor sites

```bash
python src/predict_XGBoost.py \
    -test test_rice.tsv \
    -model 'kuleshov-group/PlantCaduceus_l20' \
    -classifier classifiers/PlantCaduceus_l20/TIS_XGBoost.json \
    -device 'cuda:0' \
    -output ./output
```

**Expected output**: Predictions with confidence scores for each sequence in your test data.

## Development and Training

### Pre-training PlantCAD

For advanced users who want to pre-train PlantCAD or PlantCAD2 models from scratch or fine-tune on custom datasets.

**Requirements:**
- Large computational resources (multi-GPU recommended)
- WandB account for experiment tracking
- Custom genomic dataset in HuggingFace format

**Basic pre-training command:**
```bash
WANDB_PROJECT=PlantCAD python src/HF_pre_train.py \
    --do_train \
    --report_to wandb \
    --prediction_loss_only True \
    --remove_unused_columns False \
    --dataset_name 'kuleshov-group/Angiosperm_16_genomes' \
    --soft_masked_loss_weights_train 0.1 \
    --soft_masked_loss_weights_evaluation 0.0 \
    --weight_decay 0.01 \
    --optim adamw_torch \
    --dataloader_num_workers 16 \
    --preprocessing_num_workers 16 \
    --seed 32 \
    --save_strategy steps \
    --save_steps 1000 \
    --evaluation_strategy steps \
    --eval_steps 1000 \
    --logging_steps 10 \
    --max_steps 120000 \
    --warmup_steps 1000 \
    --save_total_limit 20 \
    --learning_rate 2E-4 \
    --lr_scheduler_type constant_with_warmup \
    --run_name test \
    --overwrite_output_dir \
    --output_dir "PlantCaduceus_train_1" \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --tokenizer_name 'kuleshov-group/PlantCaduceus_l20' \
    --config_name 'kuleshov-group/PlantCaduceus_l20' \
    --log_level info
```

**Key parameters:**
- `dataset_name`: Your custom dataset or use our Angiosperm dataset
- `max_steps`: Total training steps (adjust based on dataset size)
- `learning_rate`: 2E-4 works well for most cases
- Batch sizes: Adjust based on your GPU memory

### Performance benchmarks
The inference speed is highly dependent on the model size and GPU type. Performance with 5,000 SNPs:

<table>
    <tr>
        <th>Model</th>
        <th>H100</th>
        <th>A100</th>
        <th>A6000</th>
        <th>3090</th>
        <th>A5000</th>
        <th>A40</th>
        <th>2080</th>
    </tr>
    <tr>
        <td>PlantCaduceus_l20</td>
        <td>16s</td>
        <td>19s</td>
        <td>24s</td>
        <td>25s</td>
        <td>25s</td>
        <td>26s</td>
        <td>44s</td>
    </tr>
    <tr>
        <td>PlantCaduceus_l24</td>
        <td>21s</td>
        <td>27s</td>
        <td>35s</td>
        <td>37s</td>
        <td>42s</td>
        <td>38s</td>
        <td>71s</td>
    </tr>
    <tr>
        <td>PlantCaduceus_l28</td>
        <td>31s</td>
        <td>43s</td>
        <td>62s</td>
        <td>69s</td>
        <td>77s</td>
        <td>67s</td>
        <td>137s</td>
    </tr>
    <tr>
        <td>PlantCaduceus_l32</td>
        <td>47s</td>
        <td>66s</td>
        <td>94s</td>
        <td>116s</td>
        <td>130s</td>
        <td>107s</td>
        <td>232s</td>
    </tr>
</table>


## Citation
Zhai, J., Gokaslan, A., Schiff, Y., Berthel, A., Liu, Z. Y., Lai, W. L., Miller, Z. R., Scheben, A., Stitzer, M. C., Romay, M. C., Buckler, E. S., & Kuleshov, V. (2025). Cross-species modeling of plant genomes at single nucleotide resolution using a pretrained DNA language model. Proceedings of the National Academy of Sciences, 122(24), e2421738122. https://doi.org/10.1073/pnas.2421738122
