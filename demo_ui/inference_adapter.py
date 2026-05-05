import base64
import io
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

import numpy as np
import PIL.Image
import requests
import torch

from generate_image import load_generator_for_inference


class InferenceBackendMode(str, Enum):
    LOCAL = "local"
    REMOTE = "remote"


@dataclass(frozen=True)
class CheckpointPreset:
    name: str
    stage1_checkpoint: str = ""
    final_checkpoint: str = ""
    remote_endpoint: str = ""
    description: str = ""


@dataclass
class InferenceResult:
    input_image: PIL.Image.Image
    binary_mask: PIL.Image.Image
    masked_input_image: PIL.Image.Image
    stage1_image: PIL.Image.Image
    final_image: PIL.Image.Image
    pipeline_notes: Dict[str, str]


_GENERATOR_CACHE: Dict[tuple, torch.nn.Module] = {}


def pil_to_base64(image: PIL.Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def base64_to_pil(payload: str) -> PIL.Image.Image:
    return PIL.Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")


def ensure_rgb(image: PIL.Image.Image) -> PIL.Image.Image:
    return image.convert("RGB")


def ensure_binary_mask(mask_image: PIL.Image.Image, size) -> PIL.Image.Image:
    mask = mask_image.convert("L").resize(size)
    mask_np = np.array(mask, dtype=np.uint8)
    binary = np.where(mask_np >= 127, 255, 0).astype(np.uint8)
    return PIL.Image.fromarray(binary, mode="L")


def apply_mask_preview(image: PIL.Image.Image, mask_image: PIL.Image.Image) -> PIL.Image.Image:
    image_np = np.array(ensure_rgb(image), dtype=np.uint8)
    mask_np = np.array(mask_image.convert("L"), dtype=np.uint8)
    masked = image_np.copy()
    masked[mask_np > 0] = 255
    return PIL.Image.fromarray(masked, mode="RGB")


def image_to_tensor(image: PIL.Image.Image, device: torch.device) -> torch.Tensor:
    image_np = np.array(ensure_rgb(image), dtype=np.float32).transpose(2, 0, 1)
    return (torch.from_numpy(image_np).to(device) / 127.5 - 1.0).unsqueeze(0)


def mask_to_tensor(mask_image: PIL.Image.Image, device: torch.device) -> torch.Tensor:
    mask_np = np.array(mask_image.convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(mask_np).to(device).unsqueeze(0).unsqueeze(0)


def output_tensor_to_pil(output: torch.Tensor) -> PIL.Image.Image:
    output = (output.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
    return PIL.Image.fromarray(output[0].cpu().numpy(), mode="RGB")


def get_generator(network_pkl: str, device: torch.device):
    cache_key = (network_pkl, device.type)
    if cache_key not in _GENERATOR_CACHE:
        generator, _generator_key, _network_data = load_generator_for_inference(network_pkl, device)
        _GENERATOR_CACHE[cache_key] = generator
    return _GENERATOR_CACHE[cache_key]


def run_generator_on_inputs(
    network_pkl: str,
    image: PIL.Image.Image,
    mask_image: PIL.Image.Image,
    device: torch.device,
    truncation_psi: float = 1.0,
    noise_mode: str = "const",
) -> PIL.Image.Image:
    generator = get_generator(network_pkl, device)
    image_tensor = image_to_tensor(image, device)
    mask_tensor = mask_to_tensor(mask_image, device)
    label = torch.zeros([1, generator.c_dim], device=device)
    z = torch.zeros([1, generator.z_dim], device=device)

    with torch.no_grad():
        output = generator(
            image_tensor,
            mask_tensor,
            z,
            label,
            truncation_psi=truncation_psi,
            noise_mode=noise_mode,
        )
    return output_tensor_to_pil(output)


def run_local_inference(
    image: PIL.Image.Image,
    mask_image: PIL.Image.Image,
    preset: CheckpointPreset,
    device_name: Optional[str] = None,
) -> InferenceResult:
    if not preset.final_checkpoint:
        raise ValueError("A final checkpoint path is required for local inference.")

    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    input_image = ensure_rgb(image)
    binary_mask = ensure_binary_mask(mask_image, input_image.size)
    masked_input_image = apply_mask_preview(input_image, binary_mask)

    stage1_checkpoint = preset.stage1_checkpoint or preset.final_checkpoint
    stage1_image = run_generator_on_inputs(stage1_checkpoint, input_image, binary_mask, device)
    final_image = run_generator_on_inputs(preset.final_checkpoint, input_image, binary_mask, device)

    stage1_note = "Uses the Stage 1 checkpoint path." if preset.stage1_checkpoint else "Uses the final checkpoint as a fallback Stage 1 preview."

    return InferenceResult(
        input_image=input_image,
        binary_mask=binary_mask,
        masked_input_image=masked_input_image,
        stage1_image=stage1_image,
        final_image=final_image,
        pipeline_notes={
            "backend": f"Local inference on {device.type}",
            "stage1": stage1_note,
            "final": "Uses the final checkpoint for the completed restoration result.",
        },
    )


def run_remote_inference(
    image: PIL.Image.Image,
    mask_image: PIL.Image.Image,
    preset: CheckpointPreset,
    timeout_seconds: int = 120,
) -> InferenceResult:
    if not preset.remote_endpoint:
        raise ValueError("A remote endpoint is required for remote inference.")

    input_image = ensure_rgb(image)
    binary_mask = ensure_binary_mask(mask_image, input_image.size)
    masked_input_image = apply_mask_preview(input_image, binary_mask)

    response = requests.post(
        preset.remote_endpoint,
        json={
            "image_base64": pil_to_base64(input_image),
            "mask_base64": pil_to_base64(binary_mask.convert("RGB")),
            "stage1_checkpoint": preset.stage1_checkpoint,
            "final_checkpoint": preset.final_checkpoint,
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()

    return InferenceResult(
        input_image=input_image,
        binary_mask=binary_mask,
        masked_input_image=masked_input_image,
        stage1_image=base64_to_pil(payload["stage1_image"]),
        final_image=base64_to_pil(payload["final_image"]),
        pipeline_notes=payload.get(
            "pipeline_notes",
            {
                "backend": "Remote inference",
                "stage1": "Remote service returned the Stage 1 intermediate result.",
                "final": "Remote service returned the final restoration result.",
            },
        ),
    )


def run_inference(
    backend_mode: InferenceBackendMode,
    image: PIL.Image.Image,
    mask_image: PIL.Image.Image,
    preset: CheckpointPreset,
) -> InferenceResult:
    if backend_mode == InferenceBackendMode.LOCAL:
        return run_local_inference(image=image, mask_image=mask_image, preset=preset)
    if backend_mode == InferenceBackendMode.REMOTE:
        return run_remote_inference(image=image, mask_image=mask_image, preset=preset)
    raise ValueError(f"Unsupported backend mode: {backend_mode}")
