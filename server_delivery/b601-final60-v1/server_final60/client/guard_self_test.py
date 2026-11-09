"""Offline synthetic action-guard acceptance. No model, USB or robot is opened."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from action_guard import ActionGuard, GuardError, JOINTS, LOCAL_CHECKS, configuration_issues


class Clock:
    value = 100.0

    def __call__(self):
        return self.value


def run_tests():
    checks = []
    template = json.loads((Path(__file__).resolve().parent / 'robot_deployment_template.json').read_text())
    assert configuration_issues(template)

    def expect_rejected(label, operation, guard):
        try:
            operation()
        except GuardError:
            assert not guard.armed, label
            checks.append(label)
        else:
            raise AssertionError(label + ' was not rejected')

    def fresh():
        clock = Clock()
        config = {'joint_names': list(JOINTS), 'action_space': 'absolute_adapter_input', 'control_hz': 25,
            'limits': {'lower': [-1.0]*7, 'upper': [1.0]*7, 'max_velocity': [2.0]*7},
            'max_observation_age_seconds': 0.2, 'calibration_reference': 'SYNTHETIC_TEST_ONLY',
            'local_checks': {key: True for key in LOCAL_CHECKS}}
        guard = ActionGuard(config, clock=clock)
        guard.arm(np.zeros(7), sent_at=100, emergency_stop=False, operator_confirmed=True)
        clock.value = 100.04
        return guard, clock

    clock = Clock()
    guard = ActionGuard(template, clock=clock)
    expect_rejected('unconfigured_template_cannot_arm', lambda: guard.arm(
        np.zeros(7), sent_at=100, emergency_stop=False, operator_confirmed=True), guard)
    assert configuration_issues([]) and configuration_issues({'limits': None, 'local_checks': None})
    checks.append('malformed_configuration_is_not_ready')

    for label, action, observed_at, stop in [
        ('nan_action', [float('nan')]+[0.0]*6, 100.02, False),
        ('wrong_action_shape', [0.0]*6, 100.02, False),
        ('outside_limits', [1.1]*7, 100.02, False),
        ('excess_slew', [0.2]*7, 100.02, False),
        ('stale_observation', [0.0]*7, 99.0, False),
        ('future_observation', [0.0]*7, 100.05, False),
        ('emergency_stop', [0.0]*7, 100.02, True),
        ('unknown_stop_state', [0.0]*7, 100.02, None),
        ('boolean_action', [True]+[0.0]*6, 100.02, False),
        ('string_action', ['0.01']*7, 100.02, False),
        ('complex_action', [0.01+0j]*7, 100.02, False),
    ]:
        guard, clock = fresh()
        expect_rejected(label, lambda: guard.check_next(action, observed_at=observed_at, emergency_stop=stop), guard)
        np.testing.assert_array_equal(guard.last_sent_action, np.zeros(7))

    guard, clock = fresh()
    first = guard.check_next([0.04]*7, observed_at=100.02, emergency_stop=False)
    np.testing.assert_array_equal(guard.last_sent_action, np.zeros(7))
    actual = guard.before_send(first, emergency_stop=False)
    try:
        actual[0] = 0.5
        raise AssertionError('Returned send values must be immutable')
    except TypeError:
        checks.append('immutable_send_values')
    guard.commit_sent(first, actual, accepted=True, sent_at=clock.value)
    clock.value = 100.08
    second = guard.check_next([0.08]*7, observed_at=100.02, emergency_stop=False)
    actual = guard.before_send(second, emergency_stop=False)
    guard.commit_sent(second, actual, accepted=True, sent_at=clock.value)
    checks.extend(['approval_does_not_commit', 'acknowledged_send_commits', 'same_chunk_capture_time_allowed'])
    expect_rejected('replayed_approval_rejected', lambda: guard.before_send(second, emergency_stop=False), guard)

    for label in ('usb_failure_does_not_commit', 'changed_accepted_command', 'unclaimed_send',
                  'approval_replacement', 'double_send_claim', 'expiry_before_send', 'idle_time_does_not_allow_burst'):
        guard, clock = fresh()
        approval = guard.check_next([0.04]*7, observed_at=100.02, emergency_stop=False)
        if label == 'unclaimed_send':
            operation = lambda: guard.commit_sent(approval, [0.04]*7, accepted=True, sent_at=clock.value)
        elif label == 'approval_replacement':
            operation = lambda: guard.before_send(replace(approval, action=(0.05,)*7), emergency_stop=False)
        elif label == 'expiry_before_send':
            clock.value = 100.3
            operation = lambda: guard.before_send(approval, emergency_stop=False)
        elif label == 'idle_time_does_not_allow_burst':
            guard.disarm()
            guard.arm(np.zeros(7), sent_at=clock.value, emergency_stop=False, operator_confirmed=True)
            clock.value += 0.15
            operation = lambda: guard.check_next([0.2]*7, observed_at=clock.value, emergency_stop=False)
        else:
            actual = guard.before_send(approval, emergency_stop=False)
            if label == 'usb_failure_does_not_commit':
                operation = lambda: guard.commit_sent(approval, actual, accepted=False, sent_at=clock.value)
            elif label == 'changed_accepted_command':
                operation = lambda: guard.commit_sent(approval, [0.05]*7, accepted=True, sent_at=clock.value)
            else:
                operation = lambda: guard.before_send(approval, emergency_stop=False)
        expect_rejected(label, operation, guard)
        np.testing.assert_array_equal(guard.last_sent_action, np.zeros(7))

    guard, clock = fresh()
    clock.value = 99.9
    expect_rejected('backwards_send_clock', lambda: guard.check_next(
        [0.0]*7, observed_at=99.8, emergency_stop=False), guard)
    guard, clock = fresh()
    approval = guard.check_next([0.04]*7, observed_at=100.03, emergency_stop=False)
    actual = guard.before_send(approval, emergency_stop=False)
    guard.commit_sent(approval, actual, accepted=True, sent_at=clock.value)
    clock.value = 100.08
    expect_rejected('backwards_observation', lambda: guard.check_next(
        [0.08]*7, observed_at=100.02, emergency_stop=False), guard)
    assert not guard.armed
    expect_rejected('failure_stays_disarmed', lambda: guard.check_next(
        [0.04]*7, observed_at=clock.value, emergency_stop=False), guard)

    guard, clock = fresh()
    approval = guard.check_next([0.04]*7, observed_at=100.02, emergency_stop=False)
    clock.value = 100.08
    actual = guard.before_send(approval, emergency_stop=False)
    expect_rejected('ack_before_actual_claim', lambda: guard.commit_sent(
        approval, actual, accepted=True, sent_at=100.05), guard)
    np.testing.assert_array_equal(guard.last_sent_action, np.zeros(7))

    for label, values in [('bool_limits', [True]*7), ('string_limits', ['1.0']*7),
                           ('complex_limits', [1+0j]*7), ('overflow_limits', [10**1000]*7)]:
        guard, clock = fresh()
        config = dict(guard._config)
        config['limits'] = dict(config['limits'], upper=values)
        assert configuration_issues(config), label
        checks.append(label)
    for label, values, sent_at in [('stale_initial_ack', [0.0]*7, 99.0),
        ('future_initial_ack', [0.0]*7, 100.05), ('out_of_bounds_initial', [2.0]*7, 100.04)]:
        guard, clock = fresh()
        guard.disarm()
        expect_rejected(label, lambda: guard.arm(values, sent_at=sent_at,
            emergency_stop=False, operator_confirmed=True), guard)
    return checks, template


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    checks, template = run_tests()
    root = Path(__file__).resolve().parent
    report = {'status': 'passed', 'cases': checks, 'case_count': len(checks),
        'configuration_is_synthetic_test_only': True, 'default_template_cannot_arm': True,
        'real_hardware_tested': False, 'guard_can_stop_already_moving_hardware': False,
        'default_template_issues': configuration_issues(template),
        'guard_source_sha256': hashlib.sha256((root/'action_guard.py').read_bytes()).hexdigest(),
        'template_sha256': hashlib.sha256((root/'robot_deployment_template.json').read_bytes()).hexdigest(),
        'scope': 'CPU software rejection/commit semantics only, no USB/robot/Windows execution'}
    if args.output:
        assert not args.output.exists(), args.output
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
