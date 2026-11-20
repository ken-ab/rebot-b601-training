"""Optional local command guard. It validates commands; it never sends or stops hardware."""
import copy
from dataclasses import dataclass
import math
import numbers
import secrets
import threading
import time

import numpy as np

JOINTS = ('shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos', 'wrist_flex.pos',
          'wrist_yaw.pos', 'wrist_roll.pos', 'gripper.pos')
LOCAL_CHECKS = ('usb_device_and_calibration_confirmed', 'joint_order_and_units_confirmed',
                'hardware_limits_confirmed', 'camera_views_confirmed', 'physical_stop_verified')


class GuardError(ValueError):
    pass


def _numeric_vector(value, dtype):
    try:
        original = np.asarray(value, dtype=object)
        if original.shape != (7,) or any(
            isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Real) for v in original.flat
        ):
            raise GuardError('Expected seven real numeric values, not bool/string/complex')
        with np.errstate(over='ignore', invalid='ignore'):
            array = np.asarray(value, dtype=dtype)
        if not np.isfinite(array).all():
            raise GuardError('Expected seven finite numeric values')
        return array.copy()
    except (TypeError, ValueError, OverflowError) as error:
        raise GuardError('Invalid seven-axis numeric vector') from None


def _vector(value):
    return _numeric_vector(value, np.float32)


def _timestamp(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise GuardError('Timestamp must be a real local monotonic number')
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        raise GuardError('Invalid timestamp') from None
    if not math.isfinite(value):
        raise GuardError('Timestamp must be finite')
    return value


def configuration_issues(config):
    if not isinstance(config, dict):
        return ['Configuration must be a JSON object']
    issues = []
    if config.get('joint_names') != list(JOINTS):
        issues.append('Confirm exact seven-axis joint order')
    if config.get('action_space') != 'absolute_adapter_input':
        issues.append('Use absolute_adapter_input command coordinates')
    try:
        valid_rate = _timestamp(config.get('control_hz')) == 25
    except GuardError:
        valid_rate = False
    if not valid_rate:
        issues.append('Confirm the dataset-compatible25Hz command schedule')
    vectors = {}
    limits = config.get('limits') if isinstance(config.get('limits'), dict) else {}
    for key in ('lower', 'upper', 'max_velocity'):
        try:
            value = _numeric_vector(limits.get(key), np.float64)
            vectors[key] = value
        except (TypeError, ValueError, OverflowError):
            issues.append('Provide calibrated seven-axis ' + key)
    if len(vectors) == 3:
        if not np.all(vectors['lower'] < vectors['upper']):
            issues.append('Each lower limit must be below its upper limit')
        if not np.all(vectors['max_velocity'] > 0):
            issues.append('Each command-space velocity limit must be positive')
    age = config.get('max_observation_age_seconds')
    try:
        valid_age = _timestamp(age) > 0
    except GuardError:
        valid_age = False
    if not valid_age:
        issues.append('Set an explicit finite positive observation-age limit')
    if not isinstance(config.get('calibration_reference'), str) or not config['calibration_reference'].strip():
        issues.append('Record the actual local calibration reference')
    checks = config.get('local_checks') if isinstance(config.get('local_checks'), dict) else {}
    for key in LOCAL_CHECKS:
        if checks.get(key) is not True:
            issues.append('Local verification pending: ' + key)
    return issues


@dataclass(frozen=True)
class ApprovedCommand:
    token: str
    action: tuple
    observed_at: float
    checked_at: float


class ActionGuard:
    def __init__(self, config, *, clock=time.monotonic):
        self._config = copy.deepcopy(config)
        self._clock = clock
        self._lock = threading.RLock()
        self._armed = False
        self._pending = None
        self._claimed = False
        self._claimed_at = None
        self._last_action = None
        self._last_sent_at = None
        self._last_observed_at = None
        self.last_rejection = 'Not armed'

    @property
    def armed(self):
        with self._lock:
            return self._armed

    @property
    def last_sent_action(self):
        with self._lock:
            return None if self._last_action is None else self._last_action.copy()

    def disarm(self, reason='operator disarm'):
        with self._lock:
            self._armed = False
            self._pending = None
            self._claimed = False
            self._claimed_at = None
            self.last_rejection = str(reason)

    def _reject(self, reason):
        self.disarm(reason)
        raise GuardError(reason)

    def _now(self):
        value = float(self._clock())
        if not math.isfinite(value):
            self._reject('Local monotonic clock is not finite')
        return value

    def _stop_check(self, emergency_stop):
        if type(emergency_stop) is not bool or emergency_stop:
            self._reject('Emergency stop active or its state is unknown')

    def _within_limits(self, action):
        if np.any(action < self._lower) or np.any(action > self._upper):
            self._reject('Command exceeds calibrated adapter-space bounds')

    def arm(self, initial_accepted_command, *, sent_at, emergency_stop, operator_confirmed):
        with self._lock:
            try:
                if self._armed or operator_confirmed is not True:
                    self._reject('Arming requires explicit local operator confirmation')
                issues = configuration_issues(self._config)
                if issues:
                    self._reject('; '.join(issues))
                self._stop_check(emergency_stop)
                self._lower = np.asarray(self._config['limits']['lower'], dtype=np.float64)
                self._upper = np.asarray(self._config['limits']['upper'], dtype=np.float64)
                self._velocity = np.asarray(self._config['limits']['max_velocity'], dtype=np.float64)
                self._max_age = float(self._config['max_observation_age_seconds'])
                action, now = _vector(initial_accepted_command), self._now()
                sent_at = _timestamp(sent_at)
                if not math.isfinite(sent_at) or sent_at > now or now-sent_at > self._max_age:
                    self._reject('Initial command must be recently acknowledged on the local clock')
                self._within_limits(action)
                self._last_action, self._last_sent_at = action, sent_at
                self._last_observed_at, self._pending, self._claimed = None, None, False
                self._claimed_at = None
                self._armed = True
            except Exception:
                self.disarm('Arming validation failed')
                raise

    def _validate(self, action, observed_at, emergency_stop):
        if not self._armed:
            self._reject('Guard is disarmed')
        self._stop_check(emergency_stop)
        now, observed_at = self._now(), _timestamp(observed_at)
        if not math.isfinite(observed_at) or observed_at > now or now-observed_at > self._max_age:
            self._reject('Observation is stale, future-dated or from an incompatible clock')
        if self._last_observed_at is not None and observed_at < self._last_observed_at:
            self._reject('Observation capture time moved backwards')
        action = _vector(action)
        self._within_limits(action)
        elapsed = now-self._last_sent_at
        if not math.isfinite(elapsed) or elapsed <= 0:
            self._reject('Local send time must advance monotonically')
        allowed = self._velocity * min(elapsed, 1/25)
        if np.any(np.abs(action.astype(np.float64)-self._last_action) > allowed):
            self._reject('Command-space slew exceeds configured limits')
        return action, observed_at, now

    def check_next(self, action, *, observed_at, emergency_stop):
        with self._lock:
            try:
                if self._pending is not None:
                    self._reject('Previous approved command has not been resolved')
                action, observed_at, now = self._validate(action, observed_at, emergency_stop)
                approval = ApprovedCommand(secrets.token_hex(12), tuple(float(v) for v in action), observed_at, now)
                self._pending, self._claimed = approval, False
                return approval
            except Exception:
                self.disarm('Command validation failed')
                raise

    def before_send(self, approval, *, emergency_stop):
        """Call immediately before the single local USB send; no queueing afterwards."""
        with self._lock:
            try:
                if approval is not self._pending or self._claimed:
                    self._reject('Approval is unknown, already used or replaced')
                action, _, now = self._validate(approval.action, approval.observed_at, emergency_stop)
                if now < approval.checked_at:
                    self._reject('Local clock moved backwards after approval')
                self._claimed = True
                self._claimed_at = now
                return tuple(float(value) for value in action)
            except Exception:
                self.disarm('Pre-send validation failed')
                raise

    def commit_sent(self, approval, accepted_command, *, accepted, sent_at):
        """Commit only after the adapter acknowledges the same command-space values."""
        with self._lock:
            try:
                if approval is not self._pending or not self._claimed or accepted is not True:
                    self._reject('Send is unresolved, rejected or not locally acknowledged')
                actual, now, sent_at = _vector(accepted_command), self._now(), _timestamp(sent_at)
                if not np.array_equal(actual, np.asarray(approval.action, dtype=np.float32)):
                    self._reject('Adapter accepted a different command; state must be re-established')
                if not self._last_sent_at < sent_at <= now or sent_at < self._claimed_at:
                    self._reject('Invalid successful-send timestamp')
                self._last_action, self._last_sent_at = actual, sent_at
                self._last_observed_at = approval.observed_at
                self._pending, self._claimed = None, False
                self._claimed_at = None
            except Exception:
                self.disarm('Send confirmation failed')
                raise
