"""Time-indexed action chunks and executable RTC prefixes (motor degrees)."""

import math
from collections import deque
import numpy as np

RTC_KEY = "b601_rtc"
RTC_PROTOCOL = "pi05-inference-rtc-v1"


def require_rtc(metadata):
    if metadata.get(RTC_KEY) != RTC_PROTOCOL:
        raise ValueError("服务器尚未启用配套的 π0.5 RTC，请先启动更新后的推理服务。")


class DelayEstimator:
    """Paper Algorithm 1: maximum of recent observed delays, in control ticks."""

    def __init__(self, initial_ticks, window=10):
        self.history = deque([max(0, int(initial_ticks))], maxlen=window)

    def observe(self, ticks):
        self.history.append(max(0, int(ticks)))

    @property
    def ticks(self):
        return max(self.history)


def rtc_context(timeline, tick, last, estimated_delay, request_id):
    """Never guide invented holds or a prefix beyond the available overlap.

    With H < 2*d, the paper's d <= s <= H-d has no feasible window.
    Use a plain, still time-aligned request in those cases and log why.
    This selects a sampler; it never stops or restricts robot motion.
    """
    previous, overlap = timeline.prefix(tick, last)
    if 2 * estimated_delay > timeline.horizon:
        return None, "delay_exceeds_half_prediction_horizon", overlap
    if overlap <= estimated_delay:
        return None, "insufficient_valid_old_overlap", overlap
    return {
        "protocol": RTC_PROTOCOL, "request_id": request_id,
        "observation_tick": tick, "inference_delay": estimated_delay,
        "prefix_horizon": min(overlap, timeline.horizon - estimated_delay),
        "previous_actions": previous,
    }, None, overlap


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

    def install_aligned(self, actions, observation_tick, request_id, arrival_tick, *, hold_first=False):
        """Allow up to 200 ms of playout lag at 25 Hz; discard fully expired replies."""
        age = int(arrival_tick) - int(observation_tick)
        playout_delay = min(5, max(0, age))
        first_index = age - playout_delay
        if first_index + int(hold_first) >= self.horizon:
            return None
        self.install(actions, observation_tick + playout_delay, request_id)
        return first_index

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
