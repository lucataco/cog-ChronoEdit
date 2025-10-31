# ChronoEdit Cog Model

This repository packages NVIDIA's **ChronoEdit-14B Diffusers** image editing model for deployment on [Replicate](https://replicate.com) using [Cog](https://github.com/replicate/cog). ChronoEdit treats image editing as a temporal generation task to preserve structure while applying rich edits, and ships here with NVIDIA's 8-step distillation LoRA for fast inference.

## Features

- Loads the official ChronoEdit diffusers pipeline (`nvidia/ChronoEdit-14B-Diffusers`).
- Applies the provided 8-step distillation LoRA for accelerated inference.
- Optional prompt expansion via `Qwen/Qwen3-VL-8B-Instruct` (matches the NVIDIA Gradio demo).
- Caches model weights under `checkpoints/` to avoid repeated downloads during local testing.

## Requirements

- Docker (required by Cog)
- An NVIDIA GPU with at least 40 GB of VRAM is recommended. Smaller GPUs may OOM.
- Python is not required directly; Cog will manage the runtime inside Docker.

## Getting Started

Install Cog if you have not already:

```bash
curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)
chmod +x /usr/local/bin/cog
```

### Download Once (optional)

The first `cog predict` run downloads ~20 GB of model assets. To warm the cache ahead of time:

```bash
cog predict -i image=@path/to/example.png -i prompt="Describe the edit you want"
```

Weights will be stored in `checkpoints/ChronoEdit-14B-Diffusers/` so subsequent runs reuse them.

### Running a Prediction

```bash
cog predict \
  -i image=@input.png \
  -i prompt="Replace the flowers with glass sculptures" \
  -i enable_prompt_expansion=true \
  -i num_steps=8
```

Outputs are written to `output.jpg` by default. Set `enable_prompt_expansion=false` to skip the Qwen-based rewrite.

## Project Structure

- `predict.py` – Cog predictor loading ChronoEdit, LoRA weights, and optional prompt enhancer.
- `chronoedit_diffusers/` – Vendorized ChronoEdit pipeline modules referenced by the predictor.
- `requirements.txt` – Python dependencies installed inside the Cog container.
- `cog.yaml` – Runtime configuration (Python 3.11, CUDA 12.4 base image, GPU enabled).

## Replicate Deployment

After testing locally:

```bash
cog login
cog push r8.im/<username>/chronoedit
```

Replace `<username>` with your Replicate handle. Follow Replicate's CLI prompts to publish the model card.

## Notes

- The base image uses CUDA 12.4; adjust `cog.yaml` if your target environment requires a different CUDA version.
- Prompt expansion loads a large vision-language model; disable it to reduce memory usage or cold-start time.
- Grande-scale edits remain computationally intensive despite distillation—expect multi-minute cold starts.

For more details on ChronoEdit itself, see NVIDIA's [ChronoEdit model card](https://huggingface.co/nvidia/ChronoEdit-14B-Diffusers) and [project page](https://research.nvidia.com/labs/toronto-ai/chronoedit/).

