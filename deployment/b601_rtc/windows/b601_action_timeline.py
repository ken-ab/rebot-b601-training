"""Time-indexed action chunks and executable RTC prefixes (motor degrees)."""

import math
import numpy as np

RTC_KEY = "b601_rtc"
RTC_PROTOCOL = "pi05-inference-rtc-v1"


def require_rtc(metadata):
    if metadata.get(RTC_KEY) != RTC_PROTOCOL:
        raise ValueError("服务器尚未启用配套的 π0.5 RTC，请先启动更新后的推理服务。")


class ActionTimeline:
    """An action at index k belongs to observation_tick+k, never arrival+k."""

    def __init__(self, direction, lower, upper, max_delta, horizon=25):
        self.direction = np.asarray(direction, dtype=float)
        self.lower, self.upper = np.asarray(lower), np.asarray(upper)
        self.max_delta = max_delta
        self.horizon = horizon
        self.origin = 0
        self.actions = np.empty((0, len(self.direction)))
        self.request_id = None

    def install(self, actions, observation_tick, request_id):
        actions = np.asarray(actions, dtype=float)
        if actions.shape != (self.horizon, len(self.direction)) or not np.isfinite(actions).all():
            raise ValueError("Invalid action chunk")
        self.actions = actions * self.direction
        self.origin, self.request_id = int(observation_tick), request_id

    def proposed(self, tick, last):
        index = int(tick) - self.origin
        if 0 <= index < len(self.actions):
            return self.actions[index].copy(), index
        return np.asarray(last).copy(), None

    def limit(self, last, proposed):
        proposed = np.clip(proposed, self.lower, self.upper)
        return np.clip(last + np.clip(proposed - last, -self.max_delta, self.max_delta),
                       self.lower, self.upper)

    def prefix(self, observation_tick, last):
        """Predict the *issued*, post-limiter old trajectory in adapter units.

        Only its remaining valid overlap gets guidance. Unknown future steps
        are hold-filled for a fixed wire shape, but are masked out on server.
        """
        count = max(0, min(self.horizon, self.origin + len(self.actions) - observation_tick))
        previous = np.asarray(last).copy()
        result = []
        for offset in range(self.horizon):
            proposed, _ = self.proposed(observation_tick + offset, previous)
            previous = self.limit(previous, proposed)
            result.append(previous / self.direction)
        return np.asarray(result, dtype=np.float32), count


def elapsed_tick(now, started, fps):
    return max(0, int(math.floor((now - started) * fps + 1e-7)))


def delay_ticks(roundtrip_ms, fps):
    """Latency estimate, in controller ticks; no fixed delay/safety threshold."""
    return max(0, int(math.ceil(roundtrip_ms * fps / 1000)))
