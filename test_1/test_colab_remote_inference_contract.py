import io
import unittest
from unittest.mock import patch

import PIL.Image
import requests

from demo_ui.inference_adapter import (
    CheckpointPreset,
    InferenceBackendMode,
    _GENERATOR_CACHE,
    get_generator,
    pil_to_base64,
    run_inference,
    run_remote_inference,
)


def image(color=(12, 34, 56)):
    return PIL.Image.new("RGB", (8, 8), color)


def mask():
    return PIL.Image.new("L", (8, 8), 255)


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError("server error")
        return None

    def json(self):
        return self.payload


class ColabRemoteInferenceContractTest(unittest.TestCase):
    def test_get_generator_accepts_single_generator_loader_return(self):
        generator = object()
        device = type("Device", (), {"type": "cpu"})()
        _GENERATOR_CACHE.clear()

        with patch("demo_ui.inference_adapter.load_generator_for_inference") as load:
            load.return_value = generator
            result = get_generator("checkpoint.pkl", device)

        self.assertIs(result, generator)

    def test_get_generator_can_allow_missing_stage1_params(self):
        generator = object()
        device = type("Device", (), {"type": "cpu"})()
        _GENERATOR_CACHE.clear()

        with patch("demo_ui.inference_adapter.load_generator_for_inference") as load:
            load.return_value = generator
            result = get_generator("stage1.pkl", device, allow_missing_params=True)

        self.assertIs(result, generator)
        load.assert_called_once_with("stage1.pkl", device, allow_missing_params=True)

    def test_remote_inference_posts_multipart_pngs_to_infer_endpoint(self):
        output = image((90, 80, 70))
        preset = CheckpointPreset(
            name="Colab",
            final_checkpoint="final",
            remote_endpoint="https://salaried-easter-epileptic.ngrok-free.dev",
        )

        with patch("demo_ui.inference_adapter.requests.post") as post:
            post.return_value = FakeResponse({"final_image_base64": pil_to_base64(output)})

            result = run_remote_inference(
                image=image(),
                mask_image=mask(),
                preset=preset,
                checkpoint="final",
            )

        self.assertEqual(result.final_image.getpixel((0, 0)), output.getpixel((0, 0)))
        post.assert_called_once()
        args, kwargs = post.call_args
        self.assertEqual(args[0], "https://salaried-easter-epileptic.ngrok-free.dev/infer")
        self.assertNotIn("json", kwargs)
        self.assertEqual(kwargs["data"]["checkpoint"], "final")
        self.assertEqual(kwargs["data"]["response_format"], "json")
        self.assertEqual(kwargs["headers"]["ngrok-skip-browser-warning"], "true")
        self.assertIn("image", kwargs["files"])
        self.assertIn("mask", kwargs["files"])
        self.assertIsInstance(kwargs["files"]["image"][1], io.BytesIO)
        self.assertIsInstance(kwargs["files"]["mask"][1], io.BytesIO)

    def test_remote_inference_includes_colab_error_body(self):
        preset = CheckpointPreset(
            name="Colab",
            final_checkpoint="final",
            remote_endpoint="https://example.ngrok-free.dev",
        )

        with patch("demo_ui.inference_adapter.requests.post") as post:
            post.return_value = FakeResponse(
                {"status": "error", "error": "cannot unpack non-iterable Generator object"},
                status_code=500,
            )

            with self.assertRaisesRegex(RuntimeError, "cannot unpack non-iterable Generator object"):
                run_remote_inference(
                    image=image(),
                    mask_image=mask(),
                    preset=preset,
                    checkpoint="final",
                )

    def test_run_inference_remote_requests_final_checkpoint_only(self):
        preset = CheckpointPreset(
            name="Colab",
            stage1_checkpoint="stage1",
            final_checkpoint="final",
            remote_endpoint="https://example.ngrok-free.dev",
        )

        with patch("demo_ui.inference_adapter.run_remote_inference") as remote:
            remote.return_value = object()
            result = run_inference(
                backend_mode=InferenceBackendMode.REMOTE,
                image=image(),
                mask_image=mask(),
                preset=preset,
            )

        self.assertIs(result, remote.return_value)
        remote.assert_called_once()
        self.assertEqual(remote.call_args.kwargs["checkpoint"], "final")


if __name__ == "__main__":
    unittest.main()
