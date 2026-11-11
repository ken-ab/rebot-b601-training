"""Replay recorded views through live RTC RPC; never opens PCAN or cameras."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from b601_action_timeline import ActionTimeline, RTC_KEY, RTC_PROTOCOL, delay_ticks, require_rtc
from b601_policy_episode import infer, MAX_JOINT_DELTA_DEG
from b601_preflight import JOINTS, official_config
from rebot_pi05 import B601WebsocketClientPolicy, HOST


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("count must be positive")
    root = Path(__file__).resolve().parent
    frame_root = root / "inference-evals/b601-frames-20261111-122456"
    images = {}
    for name in ("overhead", "wrist"):
        image = cv2.imdecode(np.frombuffer((frame_root / f"000001-{name}.jpg").read_bytes(), np.uint8), 1)
        images[f"observation.images.{name}"] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    positions = np.array(json.loads((root / "b601-start-pose.json").read_text(encoding="utf-8"))["start_motor_deg"])
    config = official_config()
    direction = np.array([config.joint_directions[j] for j in JOINTS])
    lower, upper = np.array([config.joint_limits[j] for j in JOINTS]).T
    timeline = ActionTimeline(direction, lower, upper, MAX_JOINT_DELTA_DEG)
    last = np.clip(positions, lower, upper)
    client = B601WebsocketClientPolicy(HOST, args.port)
    records = []
    try:
        metadata = client.get_server_metadata()
        require_rtc(metadata)
        actions, milliseconds = infer(client, positions, images)
        baseline_ms = milliseconds
        timeline.install(actions, 0, 0)
        tick = 0
        for request_id in range(1, args.count + 1):
            previous, overlap = timeline.prefix(tick, last)
            delay = min(25, delay_ticks(milliseconds, 25))
            context = {"protocol":RTC_PROTOCOL, "request_id":request_id,
                "observation_tick":tick, "previous_actions":previous,
                "inference_delay":delay, "prefix_horizon":max(overlap,delay)}
            actions, milliseconds = infer(client, positions, images, context)
            arrival = tick + max(1, delay_ticks(milliseconds, 25))
            for old_tick in range(tick, arrival):
                proposed,_=timeline.proposed(old_tick,last)
                last=timeline.limit(last,proposed)
            timeline.install(actions,tick,request_id)
            proposed,index=timeline.proposed(arrival,last)
            last=timeline.limit(last,proposed)
            records.append({"request_id":request_id,"roundtrip_ms":round(milliseconds,2),
                "observation_tick":tick,"arrival_tick":arrival,"first_usable_index":index,
                "delay_estimate_ticks":delay,"prefix_horizon":context["prefix_horizon"],
                "shape":list(actions.shape),"all_finite":bool(np.isfinite(actions).all())})
            tick=arrival+1
    finally:
        client.close()
    times=[r["roundtrip_ms"] for r in records]
    report={"robot_commands_sent":0,"source":"recorded RGB and saved start feedback",
        "metadata":metadata,"request_count":len(records),
        "baseline_first_call_ms":round(baseline_ms,2),
        "roundtrip_p50_ms":round(float(np.percentile(times,50)),2),
        "roundtrip_p95_ms":round(float(np.percentile(times,95)),2),
        "roundtrip_max_ms":max(times),"requests":records}
    output=root/"inference-evals/rtc-upgrade-20261111/rpc-benchmark.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!='requests'},ensure_ascii=False))
    print(output)


if __name__ == "__main__":
    main()
