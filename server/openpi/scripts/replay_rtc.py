"""Timed asynchronous WebSocket playback of saved observations; NO robot commands."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import numpy as np

from openpi_client.rtc_action_queue import QueueExhausted, RTCActionQueue
from openpi_client.websocket_client_policy import WebsocketClientPolicy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--observations", default="experiments/replay_observations.npz")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--ticks", type=int, default=600)
    p.add_argument("--hz", type=float, default=30)
    p.add_argument("--replan", type=int, default=30)
    p.add_argument("--delay", type=int, default=5)
    p.add_argument("--mode", choices=("off", "guided"), default="guided")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--output", default="experiments/replay.json")
    args = p.parse_args()
    if not 0 < args.alpha <= 1: p.error("alpha must be in (0,1]")
    if args.ticks < 1 or args.hz <= 0 or not 1 <= args.replan < 50 or not 0 <= args.delay <= 50-args.replan:
        p.error("invalid tick, frequency, replan or delay parameters")
    # Decode prerecorded images before the playback clock. NpzFile indexing
    # otherwise decompresses whole arrays inside the consumer thread per request.
    with np.load(args.observations) as archive:
        data = {key: archive[key] for key in archive.files}
    queue = RTCActionQueue()
    client = WebsocketClientPolicy(args.host, args.port)
    latencies, merge_delays, merges, targets, lateness = [], [], [], [], []
    def request():
        tick = queue.tick
        contract = queue.begin(mode=args.mode, delay=args.delay)
        contract["smoothing_alpha"] = args.alpha
        i = (tick // args.replan) % len(data["state"])
        obs = {"state": data["state"][i], "images": {k: data[k][i] for k in ("cam_high", "cam_left_wrist", "cam_right_wrist")},
               "prompt": "Insert the ethernet cable into the port.", "rtc": contract}
        return obs
    def infer(obs):
        start = time.perf_counter()
        result = client.infer(obs)
        return result, (time.perf_counter()-start)*1000
    # Bootstrap before the playback clock begins; no empty queue busy-wait.
    result, bootstrap_ms = infer(request())
    queue.merge(result)
    pending = None
    failures = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        started = time.perf_counter()
        for step in range(args.ticks):
            due = started + step / args.hz
            time.sleep(max(0, due-time.perf_counter()))
            lateness.append((time.perf_counter()-due)*1000)
            try:
                if pending is not None and pending.done():
                    result, latency = pending.result()
                    merge_delays.append(queue.tick-result["rtc_info"]["observation_tick"])
                    if not queue.merge(result): raise RuntimeError("stale response rejected")
                    merges.append(queue.last_merge)
                    latencies.append(latency)
                    pending = None
                if step and step % args.replan == 0 and pending is None:
                    pending = executor.submit(infer, request())
                targets.append(queue.pop())
            except (QueueExhausted, RuntimeError) as error:
                failures.append({"tick": queue.tick, "error": str(error)})
                break
        if pending is not None:
            pending.result()  # Finish the single in-flight call before closing transport.
    from statistics import mean
    report = {"args": vars(args), "server_metadata": client.get_server_metadata(), "bootstrap_ms": bootstrap_ms,
              "consumed_ticks": len(targets), "failures": failures, "merge_delay_ticks": merge_delays,
              "merge_records": merges,
              "request_ms": latencies, "consumer_lateness_ms": {"mean": mean(lateness),
                 "p99": float(np.percentile(lateness, 99)), "max": max(lateness)},
              "limitation": "prerecorded observations; no hardware output and no task success rate"}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.save(output.with_suffix('.actions.npy'), np.asarray(targets))
    print(json.dumps(report), flush=True)
    client._ws.close()
    if failures: raise SystemExit(1)


if __name__ == "__main__": main()
