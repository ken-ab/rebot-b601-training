"""Check the versioned RTC service before restoring/enabling the robot."""

import argparse
from b601_action_timeline import require_rtc
from rebot_pi05 import B601WebsocketClientPolicy, HOST, METADATA, JPEG_RGB, TRANSPORT_KEY, check_health


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    check_health(args.port)
    client = B601WebsocketClientPolicy(HOST, args.port)
    try:
        metadata = client.get_server_metadata()
        if metadata.get(TRANSPORT_KEY) != JPEG_RGB or any(metadata.get(k) != v for k, v in METADATA.items()):
            raise ValueError("B601 图像或动作协议不一致。")
        require_rtc(metadata)
        print("π0.5 RTC 服务已就绪；动作时间对齐、成功结束、回零夹爪连续控制已启用。")
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
