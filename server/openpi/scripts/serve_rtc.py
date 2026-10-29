"""Serve the existing Piper pi0.5 checkpoint with opt-in inference-time RTC."""
import argparse
import dataclasses
import json
import logging
import os
from pathlib import Path

import numpy as np

from openpi.policies import policy_config
from openpi.policies.rtc_policy import RTCPolicy
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.training import config

BASE = "/root/kube-user/ns100002-chenrui/cr/tqq/openpi_cobot_magic_work"
DEFAULT_CHECKPOINT = BASE + "/checkpoints/cobot_magic_finetune/pi05_cobot_magic_plug_cable_finetune/plug_cable_50k_bs32_bf16_full/49999"


def load_policy(checkpoint=None, *, alpha=1.0, guidance=10.0, steps=10, assets_base_dir=None):
    if checkpoint is None:
        checkpoint = os.environ.get("RTC_CHECKPOINT", DEFAULT_CHECKPOINT)
    if assets_base_dir is None:
        assets_base_dir = os.environ.get("RTC_ASSETS_BASE_DIR")
    train_config = config.get_config("pi05_cobot_magic_plug_cable_finetune")
    if assets_base_dir is not None:
        train_config = dataclasses.replace(train_config, assets_base_dir=assets_base_dir)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.norm_stats is None:
        raise FileNotFoundError(f"Missing normalization assets: {train_config.assets_dirs}")
    logging.info("Runtime config: %s; checkpoint: %s", config.__file__, checkpoint)
    base_policy = policy_config.create_trained_policy(
        train_config, checkpoint, norm_stats=data_config.norm_stats, sample_kwargs={"num_steps": steps},
    )
    return RTCPolicy(base_policy, smoothing_alpha=alpha, max_guidance=guidance)


def example():
    return {"state": np.zeros(14, np.float32),
            "images": {key: np.zeros((224, 224, 3), np.uint8) for key in ("cam_high", "cam_left_wrist", "cam_right_wrist")},
            "prompt": "Insert the ethernet cable into the port."}


def warmup(policy, observation):
    plain = policy.infer(observation)
    request = {"session_id": "warmup", "request_id": 0, "observation_tick": 0,
               "delay": 3, "prefix": plain["actions"][:20], "prefix_attention_horizon": 20, "mode": "guided"}
    policy.infer({**observation, "rtc": request})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=os.environ.get("RTC_CHECKPOINT", DEFAULT_CHECKPOINT))
    parser.add_argument("--assets-base-dir", default=os.environ.get("RTC_ASSETS_BASE_DIR"),
                        help="Assets base directory; contains the training-config subdirectory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--smoothing-alpha", type=float, default=1.0)
    parser.add_argument("--max-guidance", type=float, default=10.0)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--ready-file", default="experiments/rtc-ready.json")
    args = parser.parse_args()
    if not 1 <= args.steps <= 100:
        parser.error("steps must be in [1,100]")
    policy = load_policy(args.checkpoint, alpha=args.smoothing_alpha, guidance=args.max_guidance,
                         steps=args.steps, assets_base_dir=args.assets_base_dir)
    warmup(policy, example())
    ready = {**policy.metadata, "checkpoint": args.checkpoint, "host": args.host, "port": args.port,
             "steps": args.steps, "runtime_config": config.__file__, "ready": True}
    path = Path(args.ready_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ready, indent=2), encoding="utf-8")
    logging.info("RTC ready: %s", ready)
    WebsocketPolicyServer(policy, host=args.host, port=args.port, metadata=policy.metadata).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
