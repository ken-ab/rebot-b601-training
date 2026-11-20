"""Serve a trained B601 checkpoint using the original raw observation mapping."""
import argparse
import logging
from pathlib import Path

from b601_policy import load_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--denoise-steps', type=int, default=10)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    policy = load_policy(args.checkpoint, denoise_steps=args.denoise_steps)
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    WebsocketPolicyServer(policy=policy, host=args.host, port=args.port, metadata=policy.metadata).serve_forever()


if __name__ == '__main__':
    main()
