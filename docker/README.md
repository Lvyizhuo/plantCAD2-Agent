# Using PlantCAD with Docker

Using Docker provides a consistent and reproducible environment with all necessary dependencies and drivers installed.

## Workflow A: Using a Pre-built Image (Recommended)

This is the easiest way to get started. We provide pre-built images on GitHub Container Registry.

### 1. Pull the Docker Image
First, pull the pre-built image from the registry. This ensures you have the correct environment without needing to build it from source.

```bash
# Pull the desired version
docker pull ghcr.io/plantcad/plantcad:v1.1.0
```

### 2. Run the Analysis
Once the image is downloaded, you can run the analysis with a single command. This command starts the container, executes the script, and then stops, leaving the output in your local directory. This non-interactive method is ideal for scripting.

Note: Before running, ensure the necessary input files (like the reference genome) are downloaded to your local directory.
```bash
# If needed, download the reference genome first
wget -nc https://download.maizegdb.org/Zm-B73-REFERENCE-NAM-5.0/Zm-B73-REFERENCE-NAM-5.0.fa.gz
gunzip -f Zm-B73-REFERENCE-NAM-5.0.fa.gz

# Run the zero-shot scoring workflow in a single command
docker run --rm --gpus all -v "$(pwd)":/workspace -w /workspace ghcr.io/plantcad/plantcad:v1.1.0 \
    python src/zero_shot_score.py \
        -input-vcf examples/example_maize_snp.vcf \
        -input-fasta Zm-B73-REFERENCE-NAM-5.0.fa \
        -output scored_variants.vcf \
        -model 'kuleshov-group/PlantCaduceus_l32' \
        -device 'cuda:0'
```
- `--rm`: Automatically removes the container when it exits.
- `--gpus all`: Makes your NVIDIA GPUs available inside the container.
- `-v "$(pwd)":/workspace`: Mounts your current project directory into `/workspace` in the container.
- `-w /workspace`: Sets the container's starting directory to `/workspace`.

When the command finishes, the output file `scored_variants.vcf` will be present in your local project directory.

---

## Workflow B: Building the Image Locally

If you need to work with a modified environment, you can build the Docker image from the `Dockerfile` provided in this repository.

### 1. Build the Image
From the project root directory, run the `docker build` command. This example uses the `Dockerfile` for PyTorch 2.7.1 and CUDA 12.8.

```bash
# Clone the repository if you haven't already
# git clone --single-branch https://github.com/plantcad/PlantCaduceus.git
# cd PlantCaduceus

# Build the image and tag it as 'plantcad-env'
docker build \
  -f docker/python3.11-torch2.7.1-cuda12.8/Dockerfile \
  -t plantcad-env .
```

### 2. Run the Analysis
After the build is complete, run your analysis by passing the desired script as a command to the `docker run` call. This non-interactive method uses your locally-built image (`plantcad-env`).

```bash
docker run --rm --gpus all -v "$(pwd)":/workspace -w /workspace plantcad-env \
    python your_script.py [your_arguments]
```
You can replace `python your_script.py [your_arguments]` with any command you wish to execute, following the examples in Workflow A.

