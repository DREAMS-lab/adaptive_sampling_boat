#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from mavros_msgs.msg import WaypointReached
from mavros_msgs.srv import CommandLong
from mavros_msgs.srv import SetMode
from std_msgs.msg import Float32
from sensor_msgs.msg import BatteryState
from rclpy.qos import qos_profile_sensor_data

import math
import digitalio
import board

# ==========================
#           NOTE:
# Probably should make a 
# README.txt for this, but remember
# to create workspace for 
# the FT232H by writing:
# 
# export BLINKA_FT232H=1 
#
# before running this script
# ===========================



class WinchMissionNode(Node):

    def __init__(self):
        super().__init__('winch_mission_node')

        # ==================================================
        # PARAMETERS (see config/winch.yaml)
        # ==================================================

        self.declare_parameter('sonar_shift', 0.1651)
        self.declare_parameter('winch_limit_threshold', 1.0)
        self.declare_parameter('final_winch', 12.0)
        self.declare_parameter('diameter', 0.2032)
        self.declare_parameter('num_mag', 20)
        self.declare_parameter('max_speed', 0.8)
        self.declare_parameter('min_speed', 0.35)
        self.declare_parameter('waiting_time', 3)
        self.declare_parameter('error_threshold', 0.05)
        self.declare_parameter('stall_timeout_cycles', 20)
        self.declare_parameter('encoder_jump_threshold', 0.5)
        self.declare_parameter('battery_threshold', 0.3)
        self.declare_parameter('watchdog_timeout', 0.5)
        self.declare_parameter('frequency', 20)

        # ==================================================
        # MAVROS COMMAND CLIENTS
        # ==================================================

        self.command_client = self.create_client(
            CommandLong,
            '/mavros/cmd/command'
        )

        # Mode client
        self.mode_client = self.create_client(
            SetMode,
            '/mavros/set_mode'
        )

        # ==================================================
        # PUBLISHERS
        # ==================================================

        self.winched_pub = self.create_publisher(
            Float32,
            '/winch',
            10
        )

        # ==================================================
        # SUBSCRIBERS
        # ==================================================

        self.create_subscription(
            WaypointReached,
            '/mavros/mission/reached',
            self.waypoint_callback,
            10
        )

        self.create_subscription(
            Float32,
            '/ping1d/data',
            self.sonar_callback,
            10
        )
        
        self.create_subscription(
            BatteryState,
            'mavros/battery',
            self.battery_cb,
            qos_profile_sensor_data)            
        # ==================================================
        # WAYPOINT DETECTION
        # ==================================================
        
        self.last_wp = None    

        # ==================================================
        # STATE MACHINE
        # ==================================================

        self.state = "HOME_WINCH"   # ← Start with homing procedure
        self.frequency = self.get_parameter('frequency').value
        self.timer = self.create_timer(1/self.frequency,
                                       self.timer_callback)

        # ==================================================
        # BATTERY
        # ==================================================

        self.low_battery = False
        self.battery_threshold = self.get_parameter('battery_threshold').value


        # ==================================================
        # SONAR AVERAGING
        # ==================================================

        self.sonar_sum = 0
        self.sonar_count = 0
        self.sonar_shift = self.get_parameter('sonar_shift').value
        self.winch_limit_threshold = self.get_parameter('winch_limit_threshold').value

        # ==================================================
        # ENCODER VARIABLES
        # ==================================================

        self.count = 0
        self.winch_length = 0

        self.target_length = 0
        self.cap_length = 0

        self.error_threshold = self.get_parameter('error_threshold').value

        self.waiting_time = self.get_parameter('waiting_time').value

        self.final_winch = self.get_parameter('final_winch').value

        self.wait_counter = 0
        self.diameter = self.get_parameter('diameter').value
        self.num_mag = self.get_parameter('num_mag').value
        self.max_speed = self.get_parameter('max_speed').value
        self.min_speed = self.get_parameter('min_speed').value

        # ==================================================
        # MOTOR STALL / ENCODER FAILURE DETECTION
        # ==================================================

        self.last_length_check = 0.0
        self.stall_counter = 0
        self.stall_timeout_cycles = self.get_parameter('stall_timeout_cycles').value
        self.commanded_speed = 0.0
        self.min_motion_threshold = (self.diameter*math.pi)/(4*self.num_mag) # Twice the distance measured by a single magnet

        # ==================================================
        # WATCHDOG SAFETY SYSTEM
        # ==================================================

        self.last_command_time = self.get_clock().now()
        self.watchdog_timeout = self.get_parameter('watchdog_timeout').value
        self.motor_safe = True

        # Encoder anomaly detection
        self.prev_winch_length = 0
        self.encoder_jump_threshold = self.get_parameter('encoder_jump_threshold').value
	
	
        # ==================================================
        # GPIO INPUTS for the Hall Effect Sensors
        # ==================================================

        self.sensor1 = digitalio.DigitalInOut(board.C0)
        self.sensor1.direction = digitalio.Direction.INPUT

        self.sensor2 = digitalio.DigitalInOut(board.C1)
        self.sensor2.direction = digitalio.Direction.INPUT
        
        # =================================================
        # These are the limit switch GPIO INPUTS
        # =================================================
        self.overspool_button = digitalio.DigitalInOut(board.C3)
        self.overspool_button.direction = digitalio.Direction.INPUT

        self.cal_button = digitalio.DigitalInOut(board.C2)
        self.cal_button.direction = digitalio.Direction.INPUT

        self.prev_state = self.get_state()

        self.quad_table = {
            (0,1):1,(1,3):1,(3,2):1,(2,0):1,
            (0,2):-1,(2,3):-1,(3,1):-1,(1,0):-1,
        }

        self.get_logger().info("Winch Mission Node READY (Homing Enabled)")

    # ==================================================
    # PAUSE VARIABLE
    # ==================================================
        self.in_loiter = False

    # ==================================================
    # CALLBACKS
    # ==================================================
    def battery_cb(self,msg):
        self.battery_percent=msg.percentage
        
    def waypoint_callback(self, msg):
        if self.last_wp is None:
            self.last_wp = msg.wp_seq
        if msg.wp_seq != self.last_wp:
            self.get_logger().info(f"Waypoint {msg.wp_seq} reached")
            self.last_wp = msg.wp_seq
            if self.state == "IDLE" and not self.in_loiter:
                self.set_mode("AUTO.LOITER")
                self.in_loiter = True
                
                self.state = "SONAR"
                self.sonar_sum = 0
                self.sonar_count = 0

    def sonar_callback(self, msg):

        if self.state != "SONAR":
            return

        self.sonar_sum += msg.data
        self.sonar_count += 1

        if self.sonar_count >= 30: #since the sonar's subscription is 10Hz, this measures the sonar depth for 3 seconds
            self.target_depth = (self.sonar_sum / 30.0) + self.sonar_shift
            self.cap_length = self.target_depth - self.winch_limit_threshold

            if self.cap_length > self.final_winch: #The projected final depth cannot exceed the winch's limit
                self.cap_length = self.final_winch

            self.target_length = 0
            self.state = "LOWER_WINCH"
            # change later================================
            # self.cap_length = 3.0
            # ============================================

            self.sonar_sum = 0
            self.sonar_count = 0

            self.get_logger().info(
                f"Depth measured → {self.target_depth:.2f} m"
            )

    # ==================================================
    # ENCODER LOGIC
    # ==================================================

    def get_state(self):
        return (self.sensor1.value << 1) | self.sensor2.value

    def update_winch(self): #This calculates the distance of the dispensed winch

        curr = self.get_state()

        delta = self.quad_table.get(
            (self.prev_state, curr), 0
        )

        self.count += delta
        self.prev_state = curr

        self.winch_length = (
            (self.count/(4*self.num_mag)) *
            self.diameter *
            math.pi
        ) # in meters, the length of the spooled sonde 

        if self.winch_length < 0:
            self.winch_length = 0
	# Detect sudden encoder jump
        if abs(self.winch_length - self.prev_winch_length) > self.encoder_jump_threshold:
            self.get_logger().warning("Encoder anomaly detected")
            self.state = "IDLE"

        self.prev_winch_length = self.winch_length

        # ==================================================
        # STALL / ENCODER FAILURE DETECTION
        # Essentially, if the motor is spinning but the Hall Effect sensors
        # do not register a change, this should be enacted
        # ==================================================

        length_change = abs(self.winch_length - self.last_length_check)
        """

        if abs(self.commanded_speed) > 0.1:  # motor should be moving
    
            if length_change < self.min_motion_threshold:
                self.stall_counter += 1
            else:
                self.stall_counter = 0

            if self.stall_counter > self.stall_timeout_cycles:
                self.get_logger().error("STALL DETECTED: Motor spinning but encoder not changing")
                self.set_winch(0.0)
                self.state = "IDLE"
                self.stall_counter = 0
                return

        else:
            self.stall_counter = 0
        """

        self.last_length_check = self.winch_length


    # ==================================================
    # ACTUATOR COMMAND
    # ==================================================

    def set_winch(self, value):
        self.commanded_speed = value
        # Safety layer: if system flagged unsafe, force stop
        if not self.motor_safe:
            value = 0.0
        
        if not self.command_client.wait_for_service(timeout_sec=0.2):
            return

        self.last_command_time = self.get_clock().now()
        req = CommandLong.Request()

        req.command = 187
        req.param1 = float(-value) # This controls AUX1, which is what the winch is run by. This is negative when motor is flipped 
        req.param2 = float('nan') # Could control a linear actuator for launching camera with this
        req.param3 = float('nan')
        req.param4 = float('nan')
        req.param5 = float('nan')
        req.param6 = float('nan')
        req.param7 = 0.0

        self.command_client.call_async(req)

    # ==================================================
    # THE PAUSE MECHANISM
    # ==================================================

    def set_mode(self, mode_str):
        if not self.mode_client.wait_for_service(timeout_sec=0.5):
            self.get_logger().warning("Mode service not available")
            return

        req = SetMode.Request()
        req.custom_mode = mode_str

        self.mode_client.call_async(req)
        self.get_logger().info(f">>> Mode set to {mode_str}")

    # ==================================================
    # MAIN LOOP
    # ==================================================

    def timer_callback(self):
        # ==================================================
        # WATCHDOG: ROS FREEZE PROTECTION
        # ==================================================
        """
        now = self.get_clock().now()

        if (now - self.last_command_time).nanoseconds > self.watchdog_timeout * 1e9:
            self.get_logger().error("WATCHDOG TRIGGERED → ROS freeze suspected")
            self.motor_safe = False
            self.set_winch(0.0)
            self.state = "IDLE"
            return
        else:
            self.motor_safe = True
      
        """
        self.motor_safe = True
        self.update_winch()

        msg = Float32()
        msg.data = float(self.winch_length)
        self.winched_pub.publish(msg)

        # ==================================================
        # CHECK BATTERY
        # ==================================================
        
        
        if hasattr(self, "battery_percent"):
            if self.battery_percent is not None and self.battery_percent < self.battery_threshold: #if battery is below 30% stop the winch
                self.get_logger().error("LOW BATTERY - aborting winch operation")
                self.low_battery = True
                if self.low_battery:
                    if self.state in ["LOWER_WINCH", "SONAR"]:
                        self.state = "RAISE_WINCH"

        # ==================================================
        # HOMING STATE
        # ==================================================


        if self.state == "HOME_WINCH":
            self.set_winch(-1*self.min_speed)
            
            """#Timing
            if not hasattr(self, "home_timer"):
                self.home_timer = 0
            self.home_timer += 1
            if self.home_timer > self.frequency * 10:
                self.get_logger().error("Homing timeout")
                self.set_winch(0.0)
                self.state = "IDLE"
                return
            """

            if self.cal_button.value == 0:
                self.get_logger().info("Homing complete")
                self.set_winch(0.0)
                self.count = 0
                self.winch_length = 0
                self.state = "IDLE"
                self.home_timer = 0

            return

        # ==================================================
        # OTHER STATES
        # ==================================================

        if self.state == "SONAR":
            return

        # LOWER WINCH
        if self.state == "LOWER_WINCH":

            error = self.target_length - self.winch_length
            speed = max(min(error, self.max_speed), -1*self.max_speed)

            self.set_winch(speed)
            
            if self.overspool_button.value == 1:
                self.set_winch(0.0)
                self.get_logger().error("ERROR: Winch spooled too much!")
                self.state= "RAISE_WINCH"

            if abs(error) < self.error_threshold:

                self.wait_counter += 1
                self.set_winch(0.0)

                if self.wait_counter >= self.waiting_time * self.frequency: #I want it to stop for 3 seconds and take data

                    self.target_length += 1.0
                    self.wait_counter = 0

                    if self.target_length > self.cap_length:
                        self.target_length = self.cap_length
                        self.state = "BOTTOM_WAIT"
                        # This is important! For example, if the sonar_depth says the environment is 5.4m
                        # and you increase the winch length by one each time, it'll go from 4m to 5m.
                        # However, since our cap is at 4.5m (sonar_depth minus 0.5), it'll go past it without
                        # this part. Now, in such a case, it will go to the cap 

        elif self.state == "BOTTOM_WAIT":

            self.set_winch(0.0)
            self.wait_counter += 1
            if self.wait_counter > self.waiting_time * self.frequency:

                self.state = "RAISE_WINCH"
                self.wait_counter = 0

        elif self.state == "RAISE_WINCH":

            self.set_winch(self.min_speed)
            if self.cal_button.value == 0:

                self.get_logger().info("Winch surface calibrated")
                self.count = 0
                
                if self.in_loiter:
                    self.set_mode("AUTO.MISSION")
                    self.in_loiter = False
                
                self.state = "IDLE"

        elif self.state == "IDLE":

            self.set_winch(0.0)


# ==================================================

def main(args=None):

    rclpy.init(args=args)
    node = WinchMissionNode()
    #rclpy.spin(node)
    #node.destroy_node()
    #rclpy.shutdown()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt received! Stopping winch...")

    finally:
        #  CRITICAL: stop the motor
        node.set_winch(0.0)

        # Optional: give time for command to send
        import time
        time.sleep(0.2)

        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
