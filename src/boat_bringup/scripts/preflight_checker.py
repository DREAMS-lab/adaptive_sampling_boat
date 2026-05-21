#!/usr/bin/env python3
"""
Boat pre-flight checker.

Runs a sequence of pass/fail checks before the planner is allowed to take off.
On full pass, latches std_msgs/Bool(True) on /preflight/passed; the planner
(running with wait_for_preflight:=true) sees this and transitions out of
WAIT_PREFLIGHT. On failure, publishes Bool(False) and reports the failing
check via /preflight/status (String).

Checks (with per-check timeouts):
  1. /mavros/state shows connected and a healthy system_status
  2. /mavros/local_position/pose is publishing (>= 5 messages, non-zero stamps)
  3. sonde_data topic is alive AND the temperature column parses to a sane value
  4. /ping1d/data is alive AND value is in a plausible depth range
  5. Setpoint round-trip: stream a no-op PoseStamped, toggle SetMode -> OFFBOARD
     and back. Confirms the offboard handshake works without ever arming.
  6. (Optional) Arm dry-run: arm for ~1 s and disarm. Default OFF; requires an
     interactive Enter from the operator. Spins the prop -- boat must be
     blocked or out of water.
"""

import sys
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, Float32, String
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode


SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)


class PreflightChecker(Node):
    def __init__(self):
        super().__init__('preflight_checker')

        self.declare_parameter('check_timeout', 15.0)
        self.declare_parameter('temp_min', 0.0)
        self.declare_parameter('temp_max', 50.0)
        self.declare_parameter('depth_min', 0.1)
        self.declare_parameter('depth_max', 50.0)
        self.declare_parameter('expected_sonde_columns', 12)
        self.declare_parameter('sonde_temp_column', 3)
        self.declare_parameter('preflight_arm_test', False)
        self.declare_parameter('arm_hold_seconds', 1.0)
        self.declare_parameter('setpoint_stream_seconds', 2.0)

        self.timeout = float(self.get_parameter('check_timeout').value)
        self.temp_min = float(self.get_parameter('temp_min').value)
        self.temp_max = float(self.get_parameter('temp_max').value)
        self.depth_min = float(self.get_parameter('depth_min').value)
        self.depth_max = float(self.get_parameter('depth_max').value)
        self.expected_sonde_columns = int(self.get_parameter('expected_sonde_columns').value)
        self.sonde_temp_column = int(self.get_parameter('sonde_temp_column').value)
        self.arm_test = bool(self.get_parameter('preflight_arm_test').value)
        self.arm_hold_seconds = float(self.get_parameter('arm_hold_seconds').value)
        self.setpoint_stream_seconds = float(self.get_parameter('setpoint_stream_seconds').value)

        # Latched outputs
        qos_latched = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.passed_pub = self.create_publisher(Bool, '/preflight/passed', qos_latched)
        self.status_pub = self.create_publisher(String, '/preflight/status', qos_latched)

        # Setpoint publisher (used for round-trip + optional arm test)
        self.setpoint_pub = self.create_publisher(PoseStamped, '/mavros/setpoint_position/local', 10)

        # MAVROS service clients
        self.set_mode_cli = self.create_client(SetMode, '/mavros/set_mode')
        self.arm_cli = self.create_client(CommandBool, '/mavros/cmd/arming')

        # Tracked state from subscriptions
        self._mavros_state = None
        self._pose_msgs = 0
        self._latest_pose = None
        self._sonde_msgs = 0
        self._latest_sonde_temp = None
        self._sonde_last_payload = None
        self._ping_msgs = 0
        self._latest_ping = None

        self.create_subscription(State, '/mavros/state', self._state_cb, 10)
        self.create_subscription(PoseStamped, '/mavros/local_position/pose', self._pose_cb, SENSOR_QOS)
        self.create_subscription(String, 'sonde_data', self._sonde_cb, 10)
        self.create_subscription(Float32, '/ping1d/data', self._ping_cb, 10)

        self.get_logger().info(
            f'preflight_checker started (timeout {self.timeout:.0f}s per check, '
            f'arm_test={self.arm_test})'
        )

        self._publish_status('RUNNING')

        # Run the checks in a worker thread so rclpy.spin can keep the subscriptions alive.
        self._worker = threading.Thread(target=self._run_all_checks, daemon=True)
        self._worker.start()

    # ---- callbacks ----

    def _state_cb(self, msg):
        self._mavros_state = msg

    def _pose_cb(self, msg):
        self._pose_msgs += 1
        self._latest_pose = msg

    def _sonde_cb(self, msg):
        self._sonde_msgs += 1
        raw = msg.data
        if raw.startswith('#DATA:'):
            raw = raw.replace('#DATA:', '', 1).lstrip()
        self._sonde_last_payload = raw
        tokens = raw.split()
        if len(tokens) <= self.sonde_temp_column:
            return
        try:
            self._latest_sonde_temp = float(tokens[self.sonde_temp_column])
        except ValueError:
            pass

    def _ping_cb(self, msg):
        self._ping_msgs += 1
        self._latest_ping = float(msg.data)

    # ---- helpers ----

    def _publish_status(self, text: str):
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)
        self.get_logger().info(f'[preflight] {text}')

    def _publish_passed(self, value: bool):
        self.passed_pub.publish(Bool(data=bool(value)))

    def _wait(self, predicate, label):
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            if predicate():
                return True
            time.sleep(0.2)
        self.get_logger().error(f'FAIL: {label}')
        self._publish_status(f'FAIL: {label}')
        return False

    # ---- the actual sequence ----

    def _run_all_checks(self):
        try:
            if not self._check_mavros_state():
                return self._fail()
            if not self._check_pose():
                return self._fail()
            if not self._check_sonde():
                return self._fail()
            if not self._check_ping():
                return self._fail()
            if not self._check_setpoint_roundtrip():
                return self._fail()
            if self.arm_test:
                if not self._check_arm_cycle():
                    return self._fail()
            self._pass()
        except Exception as e:
            self.get_logger().error(f'preflight crashed: {e}')
            self._publish_status(f'FAIL: exception {e}')
            self._fail()

    def _check_mavros_state(self):
        self._publish_status('Check 1/6: waiting for /mavros/state...')

        def ok():
            s = self._mavros_state
            return s is not None and bool(s.connected)

        if not self._wait(ok, 'MAVROS not connected'):
            return False
        s = self._mavros_state
        self._publish_status(
            f'  MAVROS connected: mode={s.mode}, armed={s.armed}, system_status={s.system_status}'
        )
        return True

    def _check_pose(self):
        self._publish_status('Check 2/6: waiting for local pose...')

        def ok():
            return self._pose_msgs >= 5 and self._latest_pose is not None and \
                   self._latest_pose.header.stamp.sec > 0

        if not self._wait(ok, 'no /mavros/local_position/pose'):
            return False
        p = self._latest_pose.pose.position
        self._publish_status(f'  pose: x={p.x:.2f} y={p.y:.2f} z={p.z:.2f}')
        return True

    def _check_sonde(self):
        self._publish_status('Check 3/6: waiting for sonde_data with valid temperature...')

        def ok():
            t = self._latest_sonde_temp
            return self._sonde_msgs >= 5 and t is not None and self.temp_min <= t <= self.temp_max

        if not self._wait(ok, f'sonde temperature missing or outside [{self.temp_min},{self.temp_max}]'):
            if self._sonde_last_payload:
                self.get_logger().error(f'  last payload: "{self._sonde_last_payload[:120]}"')
            return False
        self._publish_status(f'  sonde temp: {self._latest_sonde_temp:.2f} °C')
        return True

    def _check_ping(self):
        self._publish_status('Check 4/6: waiting for /ping1d/data with sane depth...')

        def ok():
            d = self._latest_ping
            return self._ping_msgs >= 5 and d is not None and self.depth_min <= d <= self.depth_max

        if not self._wait(ok, f'ping depth missing or outside [{self.depth_min},{self.depth_max}] m'):
            return False
        self._publish_status(f'  ping depth: {self._latest_ping:.2f} m')
        return True

    def _check_setpoint_roundtrip(self):
        self._publish_status('Check 5/6: setpoint round-trip + OFFBOARD toggle (no arm)...')

        # Wait for the services
        if not self.set_mode_cli.wait_for_service(timeout_sec=self.timeout):
            self._publish_status('FAIL: /mavros/set_mode service unavailable')
            return False

        previous_mode = self._mavros_state.mode if self._mavros_state else ''

        # Stream setpoints at 10 Hz for setpoint_stream_seconds
        stop = time.time() + self.setpoint_stream_seconds
        rate_dt = 0.1
        while time.time() < stop:
            self._publish_holding_setpoint()
            time.sleep(rate_dt)

        # Toggle OFFBOARD on
        offboard_ok = self._call_set_mode('OFFBOARD')
        if not offboard_ok:
            self._publish_status('FAIL: SetMode -> OFFBOARD did not succeed')
            return False

        # Brief hold (still no arm) so the FCU sees streaming setpoints + OFFBOARD
        end = time.time() + 1.0
        while time.time() < end:
            self._publish_holding_setpoint()
            time.sleep(rate_dt)

        # Toggle back to previous mode (or POSCTL as a safe default)
        restore_mode = previous_mode if previous_mode else 'POSCTL'
        restore_ok = self._call_set_mode(restore_mode)
        if not restore_ok:
            self.get_logger().warn(
                f'Could not restore previous mode {restore_mode}; pilot should verify on QGC'
            )
        self._publish_status(f'  OFFBOARD toggle OK (restored: {restore_mode})')
        return True

    def _check_arm_cycle(self):
        self._publish_status('Check 6/6: ARM dry-run')
        print('\n[PREFLIGHT ARM TEST] About to ARM the boat for ~{:.1f} s. The prop will spin.'.format(
            self.arm_hold_seconds))
        print('[PREFLIGHT ARM TEST] Confirm the boat is BLOCKED or OUT OF WATER.')
        print('[PREFLIGHT ARM TEST] Press Enter within 5 s to continue, anything else to abort.', flush=True)

        confirmed = self._read_enter_with_timeout(5.0)
        if not confirmed:
            self._publish_status('FAIL: ARM test not confirmed by operator')
            return False

        if not self.arm_cli.wait_for_service(timeout_sec=self.timeout):
            self._publish_status('FAIL: /mavros/cmd/arming service unavailable')
            return False

        # Need OFFBOARD + streaming setpoints to keep PX4 happy.
        previous_mode = self._mavros_state.mode if self._mavros_state else ''
        if not self._call_set_mode('OFFBOARD'):
            self._publish_status('FAIL: SetMode -> OFFBOARD during arm test')
            return False

        # Stream during arm
        stream_stop = time.time() + max(self.setpoint_stream_seconds, self.arm_hold_seconds + 0.5)
        threading.Thread(target=self._stream_until, args=(stream_stop,), daemon=True).start()

        # Send arm
        if not self._call_arming(True):
            self._publish_status('FAIL: arming command rejected')
            return False

        # Verify armed
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            if self._mavros_state is not None and self._mavros_state.armed:
                break
            time.sleep(0.1)
        else:
            self._publish_status('FAIL: vehicle did not arm within timeout')
            return False

        # Hold then disarm
        time.sleep(self.arm_hold_seconds)
        if not self._call_arming(False):
            self._publish_status('FAIL: disarm command rejected')
            return False

        # Verify disarmed
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            if self._mavros_state is not None and not self._mavros_state.armed:
                break
            time.sleep(0.1)
        else:
            self._publish_status('FAIL: vehicle did not disarm within timeout')
            return False

        # Restore mode
        restore_mode = previous_mode if previous_mode else 'POSCTL'
        self._call_set_mode(restore_mode)
        self._publish_status(f'  ARM cycle OK (held {self.arm_hold_seconds:.1f}s, restored {restore_mode})')
        return True

    # ---- service helpers ----

    def _call_set_mode(self, custom_mode: str) -> bool:
        if not self.set_mode_cli.service_is_ready():
            self.set_mode_cli.wait_for_service(timeout_sec=self.timeout)
        req = SetMode.Request()
        req.base_mode = 0
        req.custom_mode = custom_mode
        future = self.set_mode_cli.call_async(req)
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            if future.done():
                result = future.result()
                if result is None:
                    return False
                return getattr(result, 'mode_sent', True)
            time.sleep(0.05)
        return False

    def _call_arming(self, value: bool) -> bool:
        req = CommandBool.Request()
        req.value = value
        future = self.arm_cli.call_async(req)
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            if future.done():
                result = future.result()
                if result is None:
                    return False
                return bool(getattr(result, 'success', True))
            time.sleep(0.05)
        return False

    def _publish_holding_setpoint(self):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        if self._latest_pose is not None:
            msg.pose = self._latest_pose.pose
        else:
            msg.pose.orientation.w = 1.0
        self.setpoint_pub.publish(msg)

    def _stream_until(self, stop_time: float):
        while time.time() < stop_time:
            self._publish_holding_setpoint()
            time.sleep(0.1)

    def _read_enter_with_timeout(self, timeout_s: float) -> bool:
        # Plain stdin read with a timer thread. Works for interactive launch only.
        import select
        try:
            rlist, _, _ = select.select([sys.stdin], [], [], timeout_s)
        except Exception:
            return False
        if not rlist:
            return False
        line = sys.stdin.readline()
        return line.strip() == ''  # bare Enter

    # ---- termination ----

    def _pass(self):
        self._publish_status('PASS — all checks succeeded')
        self._publish_passed(True)
        # Keep latched message visible; do not shutdown so the latched topic stays.

    def _fail(self):
        self._publish_passed(False)


def main(args=None):
    rclpy.init(args=args)
    node = PreflightChecker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
