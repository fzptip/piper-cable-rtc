"""Policy transforms for cobot_MAGIC (LeRobot v2.1, 14-dim dual-arm)."""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

STATE_DIM = 14
SERVED_ACTION_DIM = 14

CAM_HIGH_KEY = "cam_high"
CAM_LEFT_WRIST_KEY = "cam_left_wrist"
CAM_RIGHT_WRIST_KEY = "cam_right_wrist"

BASE_IMAGE_MODEL_KEY = "base_0_rgb"
LEFT_WRIST_MODEL_KEY = "left_wrist_0_rgb"
RIGHT_WRIST_MODEL_KEY = "right_wrist_0_rgb"


def make_cobot_magic_example() -> dict:
    """Creates a random input example for the cobot_MAGIC policy."""
    return {
        "state": np.ones((STATE_DIM,), dtype=np.float32),
        "images": {
            CAM_HIGH_KEY: np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            CAM_LEFT_WRIST_KEY: np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            CAM_RIGHT_WRIST_KEY: np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "Insert the ethernet cable into the port.",
    }


def _convert_image(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if np.issubdtype(img.dtype, np.floating):
        if img.size and float(np.max(img)) <= 1.0:
            img = (255 * img).astype(np.uint8)
        else:
            img = np.clip(img, 0.0, 255.0).astype(np.uint8)
    if img.ndim != 3:
        raise ValueError(f"Expected image to be rank-3, got shape {img.shape}")
    if img.shape[0] == 3 and img.shape[-1] != 3:
        img = einops.rearrange(img, "c h w -> h w c")
    if img.shape[-1] != 3:
        raise ValueError(f"Expected image to have 3 channels, got shape {img.shape}")
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    return img


@dataclasses.dataclass(frozen=True)
class CobotMagicInputs(transforms.DataTransformFn):
    """Inputs for cobot_MAGIC — raw dataset joint/gripper values, no Aloha remapping."""

    model_type: _model.ModelType = _model.ModelType.PI05

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"], dtype=np.float32).reshape(STATE_DIM)
        in_images = data["images"]
        for key in (CAM_HIGH_KEY, CAM_LEFT_WRIST_KEY, CAM_RIGHT_WRIST_KEY):
            if key not in in_images:
                raise ValueError(f"Expected images to contain {key}, got {tuple(in_images)}")

        base_image = _convert_image(in_images[CAM_HIGH_KEY])
        images = {
            BASE_IMAGE_MODEL_KEY: base_image,
            LEFT_WRIST_MODEL_KEY: _convert_image(in_images[CAM_LEFT_WRIST_KEY]),
            RIGHT_WRIST_MODEL_KEY: _convert_image(in_images[CAM_RIGHT_WRIST_KEY]),
        }
        image_masks = {name: np.True_ for name in images}

        inputs: dict = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class CobotMagicOutputs(transforms.DataTransformFn):
    """Outputs for cobot_MAGIC inference."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :SERVED_ACTION_DIM], dtype=np.float32)}
