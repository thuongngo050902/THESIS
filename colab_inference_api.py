import base64
import io
import os
import time
import traceback
from pathlib import Path
from typing import Optional

import torch
import PIL.Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from demo_ui.inference_adapter import (
    run_generator_on_inputs,
    ensure_binary_mask,
)

APP_ROOT = Path("/content/THESIS")

CHECKPOINTS = {
    "stage1": "/content/drive/MyDrive/THESIS/checkpoints/resume_phase1_from_finetune_plus_loss.pkl",
    "final": "/content/drive/MyDrive/THESIS/checkpoints/network-snapshot-000072.pkl",
    "mat_baseline": "/content/drive/MyDrive/THESIS/checkpoints/Places_512_FullData.pkl",
}

app = FastAPI(title="PARF Colab Inference API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # okay for thesis demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def read_pil_image(upload: UploadFile, mode: str = "RGB") -> PIL.Image.Image:
    data = upload.file.read()
    return PIL.Image.open(io.BytesIO(data)).convert(mode)


def pil_to_png_bytes(image: PIL.Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def pil_to_base64_png(image: PIL.Image.Image) -> str:
    return base64.b64encode(pil_to_png_bytes(image)).decode("utf-8")


def check_checkpoint_exists(name: str) -> str:
    if name not in CHECKPOINTS:
        raise ValueError(f"Unknown checkpoint '{name}'. Available: {list(CHECKPOINTS.keys())}")

    checkpoint_path = CHECKPOINTS[name]
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    return checkpoint_path


def allows_missing_checkpoint_params(name: str) -> bool:
    return name in {"stage1", "mat_baseline"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "checkpoints": {
            name: {
                "path": path,
                "exists": Path(path).exists(),
            }
            for name, path in CHECKPOINTS.items()
        },
    }


@app.post("/infer")
async def infer(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    checkpoint: str = Form("final"),
    response_format: str = Form("json"),  # json or image
):
    started = time.time()

    try:
        checkpoint_path = check_checkpoint_exists(checkpoint)

        input_image = read_pil_image(image, mode="RGB")
        mask_image = read_pil_image(mask, mode="L")

        # Match the app's expected mask preprocessing.
        mask_image = ensure_binary_mask(mask_image, input_image.size)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with torch.inference_mode():
            output_image = run_generator_on_inputs(
                checkpoint_path,
                input_image,
                mask_image,
                device,
                allow_missing_params=allows_missing_checkpoint_params(checkpoint),
            )

        elapsed = time.time() - started

        if response_format == "image":
            return Response(
                content=pil_to_png_bytes(output_image),
                media_type="image/png",
                headers={
                    "X-Inference-Seconds": str(round(elapsed, 3)),
                    "X-Checkpoint": checkpoint,
                },
            )

        return {
            "status": "ok",
            "checkpoint": checkpoint,
            "checkpoint_path": checkpoint_path,
            "device": str(device),
            "seconds": round(elapsed, 3),
            "final_image_base64": pil_to_base64_png(output_image),
        }

    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(exc),
                "traceback": traceback.format_exc()[-4000:],
            },
        )
