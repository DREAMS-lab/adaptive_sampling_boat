# Lab Check — Suspended Boat

For the bench session: boat suspended in cradle, props can spin freely, GPS antenna near a window. The goal is to clear every failure mode that can be checked on land before going to water.

Two machines:
- **Odroid** (on the boat) → MAVROS + sensors only
- **Laptop** (next to the boat) → planner, GP, RViz, QGC, commander

`ROS_DOMAIN_ID` must match on both. Default in `tools/setup_env.sh` is `42`.

---

## 0. Pre-power
- [ ] Boat secured in cradle, can't fall
- [ ] Prop area clear, no loose tools
- [ ] Kill switch in arm's reach
- [ ] RC transmitter on, in **MANUAL**, throttle at zero
- [ ] GPS antenna has sky view

---

## 1. Odroid — start sensors

SSH in and launch the tmux sensor session:

```bash
ssh user@odroid.local
cd ~/workspaces/boat_adaptive
source tools/setup_env.sh
bash tools/odroid_sensors.sh
```

Expect:
```
[setup_env] ROS_DOMAIN_ID=42  hostname=odroid  ws=/home/.../boat_adaptive
tmux 'boat' started with 3 windows: mavros, ping, sonde.
```

Attach to verify each window is healthy:
```bash
tmux a -t boat       # Ctrl-b 0/1/2 to switch windows, Ctrl-b d to detach
```

- [ ] **mavros window**: shows `CON: Got HEARTBEAT, connected. FCU: PX4 Autopilot`
- [ ] **ping window**: no errors, no `Failed to initialize Ping!`
- [ ] **sonde window**: shows a continuous stream of `DATA  TIME  VOID  Temp  pH  …` rows

If any window failed, fix it before continuing (usually wrong `/dev/ttyUSB*` ordering — check `ls /dev/ttyUSB*` on the Odroid and replug in MAVROS / Ping / Sonde / Winch order).

---

## 2. Laptop — link check

```bash
cd ~/workspaces/boat_adaptive
source tools/setup_env.sh
bash tools/check_link.sh
```

Expect:
```
ROS_DOMAIN_ID=42  host=zenlime
-----
1. Topic discovery...
  ok   /mavros/state
  ok   /mavros/local_position/pose
  ok   /ping1d/data
  ok   sonde_data

2. Topic rates (5 s sample each)...
  ok   /mavros/state @ ~1.0 Hz (>= 0.5)
  ok   /mavros/local_position/pose @ ~30 Hz (>= 5)
  ok   /ping1d/data @ ~10 Hz (>= 5)
  ok   sonde_data @ ~10 Hz (>= 5)

3. MAVROS state...
  connected=true  mode=POSCTL  armed=False

4. Sample values...
  pose: x: 0.04, y: -0.12, z: 0.01
  ping depth: 0.42 m
  parsed sonde temp (col 3): 21.50 °C
  ok   temp in [0,50] °C

ALL CHECKS PASS
```

- [ ] All four topics discovered (no `miss` lines)
- [ ] All four rates above minimum
- [ ] `connected=true`
- [ ] `/mavros/local_position/pose` actually publishes (this is the GPS-locked check)
- [ ] Sonde temp is sane
- [ ] Ping depth is sane

### Common failures

| Symptom | Likely cause |
|---|---|
| `miss /mavros/state` | `ROS_DOMAIN_ID` mismatch, or WiFi not connecting laptop ↔ Odroid |
| `MAVROS connected=false` | Bad USB cable, wrong `/dev/ttyUSB0`, Pixhawk off |
| `no /mavros/local_position/pose in 6 s` | No GPS lock — wait, move antenna near window, or move boat closer |
| Sonde temp parses but is 0 / NaN | Sonde probe not initialized, try restarting the sonde window on Odroid |

---

## 3. QGC sanity (laptop, parallel)

Open QGroundControl. It auto-connects to the Pixhawk via UDP `:14550`.

- [ ] GPS satellites ≥ 6, HDOP < 2
- [ ] Compass calibrated (calibrate now if not — won't arm without it)
- [ ] IMU calibrated
- [ ] No persistent red preflight banners
- [ ] Battery voltage healthy
- [ ] **Motor test** in *Vehicle Setup → Power → Motors*: spin each motor individually at low throttle. Confirm direction. Hand on kill switch.

This is the only confirmation that the wiring + ESCs + props physically work. Do not skip.

---

## 4. Pose commander — verify setpoints + motor response

This is the real bench test: send setpoints manually, watch motors actually try to chase them.

```bash
ros2 run boat_bringup pose_commander.py
```

Expect:
```
[pose_commander] streaming setpoint at 10 Hz. Type "p" to inspect.
cmds: x y | here | arm | disarm | offboard | posctl | p | q
>
```

Run this sequence:

```
> p
pose=(+0.04, -0.12)  target=(+0.00, +0.00)  dist=0.13 m  mode=POSCTL  armed=False  connected=True
```
- [ ] pose updating, mode + armed shown

```
> here
target snapped to current (0.04, -0.12)

> offboard
> arm
> p
pose=(+0.04, -0.12)  target=(+0.04, -0.12)  dist=0.00 m  mode=OFFBOARD  armed=True  connected=True
```
- [ ] `mode=OFFBOARD`, `armed=True`
- [ ] **Motors idle** (target = current pose, zero error). If motors are spinning hard here, PX4 PID is wrong — diagnose before going further.

```
> 0 0
target -> (0.00, 0.00, 0.00)
```
- [ ] Motors spin to drive toward (0, 0). Direction matches the East/North geometry.

```
> 2 0          # 2 m East — motors should bear east
> 0 2          # 2 m North — motors should bear north
> 2 2          # NE diagonal
> here         # snap back to current pose, motors return to idle
> disarm
> p
armed=False
```
- [ ] Each setpoint produces sensible thrust + heading
- [ ] `disarm` stops motors immediately

```
> posctl
> q
```

### What (x, y) means
ENU meters from the local origin. PX4 sets the origin when the EKF locks (first GPS fix). So `(2, 2)` = 2 m East + 2 m North of *where the boat first locked GPS*. Note that point — it's your reference for all setpoints today.

### If anything weird happens
Hit the kill switch on the RC and run `disarm` in the commander. The pilot's RC override is always live regardless of what the laptop is doing.

---

## 5. Preflight check (no arm)

```bash
ros2 launch boat_bringup boat_preflight.launch.py \
    start_mavros:=false start_sensors:=false
```

Watch the preflight_checker output. Expect 5 of 5 to pass:

```
[preflight] Check 1/6: waiting for /mavros/state...
[preflight]   MAVROS connected: mode=POSCTL, armed=False, system_status=4
[preflight] Check 2/6: waiting for local pose...
[preflight]   pose: x=0.04 y=-0.12 z=0.01
[preflight] Check 3/6: waiting for sonde_data with valid temperature...
[preflight]   sonde temp: 21.50 °C
[preflight] Check 4/6: waiting for /ping1d/data with sane depth...
[preflight]   ping depth: 0.42 m
[preflight] Check 5/6: setpoint round-trip + OFFBOARD toggle (no arm)...
[preflight]   OFFBOARD toggle OK (restored: POSCTL)
[preflight] PASS — all checks succeeded
```

- [ ] All 5 checks pass
- [ ] **No motors spun** (no arm requested this round)

Ctrl-C to stop.

---

## 6. Preflight ARM dry-run — props will spin

Pilot finger on kill switch. Boat is in a cradle and can't go anywhere.

```bash
ros2 launch boat_bringup boat_preflight.launch.py \
    start_mavros:=false start_sensors:=false \
    preflight_arm_test:=true
```

When the checker prints:

```
[PREFLIGHT ARM TEST] About to ARM the boat for ~1.0 s. The prop will spin.
[PREFLIGHT ARM TEST] Confirm the boat is BLOCKED or OUT OF WATER.
[PREFLIGHT ARM TEST] Press Enter within 5 s to continue, anything else to abort.
```

Press **Enter**. Watch:

- [ ] Props spin briefly (~1 s)
- [ ] Props **disarm cleanly** within ~1 s of the disarm call
- [ ] Status ends with `PASS — all checks succeeded`

If props don't disarm cleanly → **kill switch immediately** and diagnose before any water test.

---

## 7. Mini mission dry-run (10 samples, small box)

Tests the entire planner control loop end to end with the boat stationary.

```bash
ros2 launch boat_bringup boat_exact.launch.py \
    start_mavros:=false start_sensors:=false \
    field_size_x:=5 field_size_y:=5 \
    waypoint_tolerance:=10.0 \
    max_samples:=10 \
    lambda_cost:=0.3
```

`waypoint_tolerance:=10` lets the stationary boat satisfy the "reached waypoint" check so the planner burns through 10 samples in a minute or so.

Expect:
- [ ] Preflight gate releases (planner transitions out of `WAIT_PREFLIGHT`)
- [ ] Planner arms + sets OFFBOARD
- [ ] In RViz: setpoint arrow visits 3 initial waypoints, then planner-chosen adaptive points
- [ ] GP reconstruction marker (CUBE_LIST on `/info_gain/reconstruction`) builds up
- [ ] Brief prop spins toward each setpoint
- [ ] After 10 samples: `MISSION COMPLETE` logged, trial directory written

Trial output:

```bash
ls data/trials/exact/boat/trial_001/
# samples.csv  decisions.csv  decisions.json  summary.json  figures/  ...
```

- [ ] `samples.csv` has 10 rows
- [ ] `summary.json` exists with `stop_reason: max_samples_reached`
- [ ] `figures/progress.png` shows the planner's view (mean, var, acquisition, trajectory, info gain, cost)

---

## 8. After-action

- [ ] Disarm via QGC if anything is still armed
- [ ] On Odroid: `tmux kill-session -t boat`
- [ ] Power down: Pixhawk off, Odroid shutdown, RC off
- [ ] Note in logbook: GPS HDOP, anything weird, mount_offset measurement once boat is in water (next session)

---

## Stop conditions — don't proceed to water if any of these:

- [ ] GPS never locks (no `/mavros/local_position/pose`)
- [ ] Compass / IMU won't calibrate
- [ ] Motor test in QGC shows wrong direction or a dead motor
- [ ] Preflight ARM cycle doesn't disarm within ~1 s
- [ ] Pose commander setpoints don't produce sensible motor response
- [ ] `ping_safety_node` doesn't trip when sonar is lifted

If all checks pass → cleared for dock test on water.
