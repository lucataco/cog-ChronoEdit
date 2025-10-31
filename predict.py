"""Cog predictor for NVIDIA ChronoEdit with 8-step distillation LoRA."""

import math
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from cog import BasePredictor, Input, Path as CogPath
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler
from huggingface_hub import snapshot_download
from transformers import CLIPProcessor, CLIPVisionModel

from chronoedit_diffusers.pipeline_chronoedit import ChronoEditPipeline
from chronoedit_diffusers.transformer_chronoedit import ChronoEditTransformer3DModel
from prompt_enhancer import enhance_prompt, load_model

MODEL_ID = "nvidia/ChronoEdit-14B-Diffusers"
LORA_FILENAME = "lora/chronoedit_distill_lora.safetensors"
PROMPT_ENHANCER_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
MAX_AREA = 720 * 1280
NUM_FRAMES = 5
CACHE_ROOT = Path(__file__).resolve().parent / "checkpoints"
MODEL_CACHE_DIR = CACHE_ROOT / "ChronoEdit-14B-Diffusers"
MODEL_URL = "https://weights.replicate.delivery/default/nvidia/ChronoEdit-14B-Diffusers/model.tar"


def _calculate_dimensions(image: Image.Image, mod_value: int) -> tuple[int, int]:
    aspect_ratio = image.height / image.width
    width = int(round(math.sqrt(MAX_AREA / aspect_ratio)))
    height = int(round(width * aspect_ratio))
    width = max(mod_value, width // mod_value * mod_value)
    height = max(mod_value, height // mod_value * mod_value)
    return width, height


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    pipe: ChronoEditPipeline
    prompt_model: Optional[torch.nn.Module]
    prompt_processor: Optional[object]

    def setup(self) -> None:
        torch.backends.cuda.matmul.allow_tf32 = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        model_path = MODEL_CACHE_DIR
        if not os.path.exists(MODEL_CACHE_DIR):
            download_weights(MODEL_URL, MODEL_CACHE_DIR)

        image_encoder = CLIPVisionModel.from_pretrained(
            model_path,
            subfolder="image_encoder",
            torch_dtype=torch.float32,
        )

        clip_processor = CLIPProcessor.from_pretrained(
            model_path,
            subfolder="image_processor",
            use_fast=False,
        )

        vae = AutoencoderKLWan.from_pretrained(
            model_path,
            subfolder="vae",
            torch_dtype=dtype,
        )

        transformer = ChronoEditTransformer3DModel.from_pretrained(
            model_path,
            subfolder="transformer",
            torch_dtype=dtype,
        )

        self.pipe = ChronoEditPipeline.from_pretrained(
            model_path,
            image_encoder=image_encoder,
            image_processor=clip_processor,
            transformer=transformer,
            vae=vae,
        )

        lora_path = model_path / LORA_FILENAME
        if not lora_path.exists():
            raise FileNotFoundError(
                f"Expected LoRA weights at {lora_path}, but the file was not found."
            )
        self.pipe.load_lora_weights(lora_path)
        self.pipe.fuse_lora(lora_scale=1.0)

        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            flow_shift=2.0,
        )
        self.pipe.to(self.device, dtype=dtype)
        self.pipe.set_progress_bar_config(disable=True)

        self.prompt_model = None
        self.prompt_processor = None

    def _ensure_prompt_enhancer(self) -> None:
        if self.prompt_model is None or self.prompt_processor is None:
            self.prompt_model, self.prompt_processor = load_model(PROMPT_ENHANCER_MODEL)

    def predict(
        self,
        image: CogPath = Input(description="Input image to edit"),
        prompt: str = Input(description="Editing instruction"),
        enable_prompt_expansion: bool = Input(
            description="Enhance prompt with Chain-of-Thought reasoning",
            default=False,
        ),
        num_steps: int = Input(
            description="Number of diffusion steps (8 recommended for distillation)",
            default=8,
            ge=4,
            le=50,
        ),
    ) -> CogPath:
        with torch.inference_mode():
            input_image = Image.open(image).convert("RGB")

            final_prompt = prompt
            if enable_prompt_expansion:
                self._ensure_prompt_enhancer()
                final_prompt = enhance_prompt(
                    str(image),
                    prompt,
                    self.prompt_model,
                    self.prompt_processor,
                )

            mod_value = (
                self.pipe.vae_scale_factor_spatial * self.pipe.transformer.config.patch_size[1]
            )
            width, height = _calculate_dimensions(input_image, mod_value)
            resized_image = input_image.resize((width, height))

            output = self.pipe(
                image=resized_image,
                prompt=final_prompt,
                height=height,
                width=width,
                num_frames=NUM_FRAMES,
                num_inference_steps=num_steps,
                guidance_scale=1.0,
                enable_temporal_reasoning=False,
                num_temporal_reasoning_steps=0,
            ).frames[0]

            final_frame = output[-1]
            final_frame = (np.clip(final_frame, 0.0, 1.0) * 255).astype("uint8")
            result_image = Image.fromarray(final_frame)

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                result_image.save(tmp.name, format="JPEG", quality=95)
                output_path = tmp.name

        return CogPath(output_path)
