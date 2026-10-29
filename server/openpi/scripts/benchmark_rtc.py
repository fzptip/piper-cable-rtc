"""Paired offline RTC benchmark; prerecorded observations are NOT a robot rollout."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from openpi.policies.rtc_policy import JOINT_INDICES, smooth_actions
from serve_rtc import load_policy, warmup

DATASET = "/root/kube-user/ns100002-chenrui/cr/wwq/DATA_SET/plug_in_cable_lerobot_v21"


def observations(root, count=32, stride=30, episodes=0, pairs_per_episode=4):
    import io
    from PIL import Image
    import pyarrow.parquet as pq
    root = Path(root)
    info = json.loads((root / "meta/info.json").read_text())
    result, provenance = [], []
    files = sorted((root / "data").glob("chunk-*/episode_*.parquet"))
    if episodes:
        files = [files[i] for i in np.linspace(0, len(files)-1, min(episodes,len(files)),dtype=int)]
    for parquet in files:
        table = pq.read_table(parquet)
        states = table["observation.state"].to_pylist()
        captures = {}
        try:
            for key in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
                feature = "observation.images." + key
                if info["features"][feature]["dtype"] == "video":
                    import cv2
                    episode = int(parquet.stem.split('_')[-1])
                    video = root / info["video_path"].format(episode_chunk=episode // info.get("chunks_size",1000),
                                                            episode_index=episode, video_key=feature)
                    cap = cv2.VideoCapture(str(video))
                    if not cap.isOpened(): raise FileNotFoundError(video)
                    captures[key] = cap
            frames = range(0, len(states), stride)
            if episodes:
                if len(states) <= stride: continue
                starts = np.linspace(0, len(states)-stride-1, pairs_per_episode, dtype=int)
                frames = sorted({f for start in starts for f in (start,start+stride)})
            for frame in frames:
                images = {}
                for key in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
                    if key in captures:
                        cap = captures[key]
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                        ok, bgr = cap.read()
                        if not ok: raise RuntimeError(f"Could not decode {parquet.stem} frame {frame}")
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    else:
                        value = table["observation.images."+key][frame].as_py()
                        source = io.BytesIO(value["bytes"]) if value.get("bytes") else root/value["path"]
                        with Image.open(source) as image: rgb = np.asarray(image.convert('RGB'))
                    if rgb.shape != (224, 224, 3): raise ValueError(f"Unexpected stored image shape: {rgb.shape}")
                    images[key] = rgb
                result.append({"state": np.asarray(states[frame], np.float32), "images": images,
                               "prompt": "Insert the ethernet cable into the port."})
                provenance.append({"episode": parquet.stem, "frame": int(frame)})
                if len(result) == count: return result, provenance
        finally:
            for cap in captures.values(): cap.release()
    raise ValueError(f"Only {len(result)} observations, requested {count}")


def summary(values):
    a = np.asarray(values, dtype=np.float64)
    return {"mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)), "max": float(a.max())}


def metrics(actions, prefix, delay):
    joints = actions[:, JOINT_INDICES]
    boundary = actions[delay, JOINT_INDICES] - prefix[delay-1, JOINT_INDICES]
    # Compare all overlap steps to the SAME old plan for each paired variant.
    overlap = actions[:len(prefix), JOINT_INDICES] - prefix[:, JOINT_INDICES]
    return {"boundary_joint_rms": float(np.sqrt(np.mean(boundary**2))),
            "overlap_joint_rms": float(np.sqrt(np.mean(overlap**2))),
            "internal_d1_rms": float(np.sqrt(np.mean(np.diff(joints[delay:], axis=0)**2))),
            "internal_d2_rms": float(np.sqrt(np.mean(np.diff(joints[delay:], n=2, axis=0)**2)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--observations", type=int, default=32)
    parser.add_argument("--episodes", type=int, default=0, help="evenly sample episodes, four time-spanning pairs each")
    parser.add_argument("--delay", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--guidance", type=float, default=10)
    parser.add_argument("--output", default="experiments/benchmark.json")
    args = parser.parse_args()
    if args.requests < 0 or args.observations < 2 or args.episodes < 0 or not 1 <= args.delay < 20:
        parser.error("requests>=0, observations>=2, episodes>=0, delay in [1,19] required")
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    obs, provenance = observations(args.dataset, args.observations, episodes=args.episodes)
    replay_file = "replay_observations.npz" if not args.episodes else "quality_observations.npz"
    np.savez_compressed(output.parent / replay_file,
                        state=np.stack([o["state"] for o in obs]),
                        **{k: np.stack([o["images"][k] for o in obs]) for k in obs[0]["images"]})
    pairs = [i for i in range(1, len(obs)) if provenance[i]["episode"] == provenance[i-1]["episode"]
             and provenance[i]["frame"] - provenance[i-1]["frame"] == 30]
    if not pairs: raise ValueError("No continuous observation pairs available")
    if args.requests == 0: args.requests = len(pairs)
    policy = load_policy(guidance=args.guidance)
    warmup(policy, obs[0])
    noise_rng = np.random.default_rng(20261016)
    noises = noise_rng.normal(size=(args.observations, 50, 32)).astype(np.float32)
    references = [policy.infer(o, noise=noises[i])["actions"] for i, o in enumerate(obs)]
    # Optional noise changes the JIT signature: warm this exact guided signature
    # as well, rather than allowing its compilation into the first timed sample.
    warm_request = {"session_id": "noise-warmup", "request_id": 0, "observation_tick": 0,
                    "delay": args.delay, "prefix": references[0][30:], "prefix_attention_horizon": 20,
                    "mode": "guided"}
    warm_started = time.perf_counter()
    policy.infer({**obs[pairs[0]], "rtc": warm_request}, noise=noises[pairs[0]])
    fixed_noise_warmup_ms = (time.perf_counter()-warm_started)*1000
    records = []
    for i in range(args.requests):
        # Warmup, variant order and random noise are controlled; do not compare
        # separate self-generated trajectories as if they had the same targets.
        index = pairs[i % len(pairs)]
        prefix = references[index-1][30:].copy()
        request = {"session_id": "offline-paired", "request_id": i,
                   "observation_tick": 30*i, "delay": args.delay, "prefix": prefix,
                   "prefix_attention_horizon": len(prefix)}
        for mode in (("off", "guided") if i % 2 == 0 else ("guided", "off")):
            started = time.perf_counter()
            result = policy.infer({**obs[index], "rtc": {**request, "mode": mode}}, noise=noises[index])
            elapsed = (time.perf_counter()-started)*1000
            actions = result["actions"]
            records.append({"request": i, "variant": mode, "wall_ms": elapsed,
                            "model_ms": result["policy_timing"]["infer_ms"], **metrics(actions, prefix, args.delay)})
            if mode == "guided":
                filter_start = time.perf_counter()
                filtered = smooth_actions(actions, args.alpha, args.delay, prefix[args.delay-1])
                filter_ms = (time.perf_counter()-filter_start)*1000
                records.append({"request": i, "variant": "guided_ema", "wall_ms": elapsed+filter_ms,
                                "model_ms": result["policy_timing"]["infer_ms"], "filter_ms": filter_ms,
                                **metrics(filtered, prefix, args.delay)})
        if (i+1) % 25 == 0:
            print(json.dumps({"progress": i+1, "total": args.requests}), flush=True)
            output.with_suffix('.partial.json').write_text(json.dumps(records), encoding="utf-8")
    summaries = {}
    for variant in ("off", "guided", "guided_ema"):
        rows = [r for r in records if r["variant"] == variant]
        summaries[variant] = {key: summary([r[key] for r in rows]) for key in rows[0] if key not in ("request", "variant")}
    import jax
    memory = jax.devices()[0].memory_stats()
    report = {"git_commit": subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
              "args": vars(args), "observations": provenance, "summary": summaries,
              "fixed_noise_guided_warmup_ms": fixed_noise_warmup_ms,
              "unique_observation_pairs": len(pairs),
              "jax_memory_stats": memory, "rows": records,
              "limitations": ["prerecorded observations; no robot feedback or task success rate",
                              "guided_ema reuses guided samples; its estimated latency adds measured filter cost",
                              "delay is a controlled experimental parameter, not a measured controller delay"]}
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summaries, "memory": memory}), flush=True)


if __name__ == "__main__": main()
