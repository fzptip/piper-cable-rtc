import dataclasses

import numpy as np

from openpi import transforms
from openpi.models import model as _model


STATE_DIM = 16
ARM_JOINT_DIM = 7
TOTAL_JOINT_DIM = ARM_JOINT_DIM * 2

HEAD_IMAGE_KEY = "observation.head_image"
LEFT_WRIST_IMAGE_KEY = "observation.left_wrist_image"
RIGHT_WRIST_IMAGE_KEY = "observation.right_wrist_image"
JOINT_POSITION_KEY = "observation.states.joint.position"
EFFECTOR_POSITION_KEY = "observation.states.effector.position"
ACTION_JOINT_POSITION_KEY = "actions.joint.position"
ACTION_EFFECTOR_POSITION_KEY = "actions.effector.position"
PROMPT_KEY = "prompt"
TASK_KEY = "task"

BASE_IMAGE_MODEL_KEY = "base_0_rgb"
LEFT_WRIST_MODEL_KEY = "left_wrist_0_rgb"
RIGHT_WRIST_MODEL_KEY = "right_wrist_0_rgb"


def make_g1_example() -> dict:
    """Creates a random input example for the G1 policy."""
    return {
        HEAD_IMAGE_KEY: np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        LEFT_WRIST_IMAGE_KEY: np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        RIGHT_WRIST_IMAGE_KEY: np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        JOINT_POSITION_KEY: np.random.rand(TOTAL_JOINT_DIM).astype(np.float32),
        EFFECTOR_POSITION_KEY: np.random.rand(2).astype(np.float32),
        PROMPT_KEY: "do something",
    }


def _as_vector(value, *, name: str, dim: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.size != dim:
        raise ValueError(f"Expected {name} to have {dim} values, got shape {array.shape}")
    return array.reshape(dim)


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        if image.size and float(np.max(image)) <= 1.0:
            image = (255 * image).astype(np.uint8)
        else:
            image = np.clip(image, 0.0, 255.0).astype(np.uint8)
    if image.ndim != 3:
        raise ValueError(f"Expected image to be rank-3, got shape {image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.transpose(image, (1, 2, 0))
    if image.shape[-1] != 3:
        raise ValueError(f"Expected image to have 3 channels, got shape {image.shape}")
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)

    return image


def _extract_images(data: dict) -> dict[str, np.ndarray]:
    return {
        "head": _parse_image(data[HEAD_IMAGE_KEY]),
        "left_wrist": _parse_image(data[LEFT_WRIST_IMAGE_KEY]),
        "right_wrist": _parse_image(data[RIGHT_WRIST_IMAGE_KEY]),
    }


def _extract_state(data: dict) -> np.ndarray:
    joint_position = _as_vector(
        data[JOINT_POSITION_KEY],
        name=JOINT_POSITION_KEY,
        dim=TOTAL_JOINT_DIM,
    )
    effector_position = _as_vector(
        data[EFFECTOR_POSITION_KEY],
        name=EFFECTOR_POSITION_KEY,
        dim=2,
    )
    return _pack_state_or_actions(joint_position, effector_position)


def _pack_state_or_actions(joint_position: np.ndarray, effector_position: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            joint_position[..., :ARM_JOINT_DIM],
            effector_position[..., :1],
            joint_position[..., ARM_JOINT_DIM:],
            effector_position[..., 1:],
        ],
        axis=-1,
    ).astype(np.float32)


def _extract_actions(data: dict) -> np.ndarray | None:
    joint_position = data.get(ACTION_JOINT_POSITION_KEY)
    effector_position = data.get(ACTION_EFFECTOR_POSITION_KEY)
    if joint_position is None and effector_position is None:
        return None
    if joint_position is None or effector_position is None:
        raise ValueError(f"Expected both `{ACTION_JOINT_POSITION_KEY}` and `{ACTION_EFFECTOR_POSITION_KEY}`.")

    joint_position = np.asarray(joint_position, dtype=np.float32)
    effector_position = np.asarray(effector_position, dtype=np.float32)
    if joint_position.ndim == 0 or joint_position.shape[-1] != TOTAL_JOINT_DIM:
        raise ValueError(
            f"Expected {ACTION_JOINT_POSITION_KEY} to have trailing dim {TOTAL_JOINT_DIM}, got shape {joint_position.shape}"
        )
    if effector_position.ndim == 0 or effector_position.shape[-1] != 2:
        raise ValueError(
            f"Expected {ACTION_EFFECTOR_POSITION_KEY} to have trailing dim 2, got shape {effector_position.shape}"
        )
    if joint_position.shape[:-1] != effector_position.shape[:-1]:
        raise ValueError(
            "Expected joint and effector actions to share batch/action-horizon dims, "
            f"got {joint_position.shape} vs {effector_position.shape}"
        )
    return _pack_state_or_actions(joint_position, effector_position)


def _parse_prompt(data: dict):
    prompt = data.get(PROMPT_KEY)
    if prompt is None:
        prompt = data.get(TASK_KEY)
    if prompt is None:
        return None
    if isinstance(prompt, bytes):
        return prompt.decode("utf-8")
    if isinstance(prompt, np.ndarray) and prompt.ndim == 0:
        prompt = prompt.item()
        if isinstance(prompt, bytes):
            return prompt.decode("utf-8")
    return prompt


@dataclasses.dataclass(frozen=True)
class G1Inputs(transforms.DataTransformFn):
    """Inputs for the G1 policy.

    Expected flat input keys:
    - `observation.head_image`
    - `observation.left_wrist_image`
    - `observation.right_wrist_image`
    - `observation.states.joint.position`
    - `observation.states.effector.position`
    - optional `actions.joint.position`
    - optional `actions.effector.position`
    - optional `prompt`
    """

    model_type: _model.ModelType = _model.ModelType.PI05

    def __post_init__(self):
        if self.model_type != _model.ModelType.PI05:
            raise ValueError(f"G1Inputs only supports PI0.5, got model_type={self.model_type}")

    def __call__(self, data: dict) -> dict:
        state = _extract_state(data)
        parsed_images = _extract_images(data)
        template_image = next(iter(parsed_images.values()))

        image_sources = {
            BASE_IMAGE_MODEL_KEY: parsed_images.get("head"),
            LEFT_WRIST_MODEL_KEY: parsed_images.get("left_wrist"),
            RIGHT_WRIST_MODEL_KEY: parsed_images.get("right_wrist"),
        }
        image_masks = {name: np.bool_(image is not None) for name, image in image_sources.items()}

        inputs = {
            "state": state,
            "image": {
                name: image if image is not None else np.zeros_like(template_image)
                for name, image in image_sources.items()
            },
            "image_mask": image_masks,
        }

        if (actions := _extract_actions(data)) is not None:
            inputs["actions"] = actions

        if (prompt := _parse_prompt(data)) is not None:
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class G1Outputs(transforms.DataTransformFn):
    """Outputs for the G1 policy."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :STATE_DIM], dtype=np.float32)}
