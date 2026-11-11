"""Run the policy entry point while archiving inference views off the control thread."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
from time import perf_counter

import cv2

import b601_policy_episode as policy


def main():
    root = Path(__file__).resolve().parent / "inference-evals" / f"b601-frames-{datetime.now():%Y%m%d-%H%M%S}"
    root.mkdir(parents=True)
    original = policy.camera_images
    capture_errors = []
    started = perf_counter()
    future = None
    sequence = 0

    def save_frames(number, seconds, images):
        try:
            record = {"sequence": number, "seconds_from_launch": seconds, "images": {}}
            for key, frame in images.items():
                path = root / f"{number:06d}-{key.rsplit('.', 1)[1]}.jpg"
                encoded_ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if not encoded_ok:
                    raise OSError(f"Could not write {path}")
                path.write_bytes(encoded.tobytes())
                record["images"][key] = path.name
            with (root / "frames.jsonl").open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as error:
            capture_errors.append(str(error))

    with ThreadPoolExecutor(max_workers=1) as writer:
        def images_with_capture(cameras):
            nonlocal future, sequence
            images = original(cameras)
            if not capture_errors and (future is None or future.done()):
                future = writer.submit(save_frames, sequence, perf_counter() - started,
                                       {key: frame.copy() for key, frame in images.items()})
                sequence += 1
            return images

        policy.camera_images = images_with_capture
        print(f"实景序列保存到：{root}", flush=True)
        try:
            result = policy.main()
        finally:
            policy.camera_images = original
    (root / "capture.json").write_text(json.dumps({"frames_submitted": sequence,
        "capture_errors": capture_errors}, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
