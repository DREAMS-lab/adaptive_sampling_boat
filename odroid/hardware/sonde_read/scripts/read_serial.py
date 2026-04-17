#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import serial
import time
import subprocess


class SondeReader(Node):
	def __init__(self):
		super().__init__('sonde_reader')

		# Parameters (see config/sensors.yaml)
		self.declare_parameter('port', '/dev/boat_sonde')
		self.declare_parameter('baud', 19200)
		self.declare_parameter('rate_hz', 10.0)
		self.declare_parameter('enable_command', 'SCR -on')
		self.declare_parameter('version_command', 'VER')

		self.port = self.get_parameter('port').value
		self.baud = int(self.get_parameter('baud').value)
		self.rate_hz = float(self.get_parameter('rate_hz').value)
		self.enable_command = self.get_parameter('enable_command').value
		self.version_command = self.get_parameter('version_command').value

		self.publisher_ = self.create_publisher(String, 'sonde_data', 10)
		self.timer = self.create_timer(1.0 / self.rate_hz, self.read_sonde_data)

		self.setup_serial_port()
		self.ser = serial.Serial(self.port, baudrate=self.baud, timeout=1)
		self.ser.write(f"{self.version_command}\r".encode())
		time.sleep(0.5)
		self.ser.write(f"{self.enable_command}\r".encode())

		self.columns = ["DATA", "TIME", "VOID", "Tempe dec C", "pH units", "Depth m",
		                "SpCond uS/cm", "HDO %Sat", "HDO mg/l", "Chl ug/l", "CDOM ppb", "Turb NTU"]
		self.column_widths = [10, 9, 9, 12, 10, 8, 14, 9, 9, 9, 10, 9]

		header = "".join(f"{name:<{width}}" for name, width in zip(self.columns, self.column_widths))
		print(header)

	def setup_serial_port(self):
		command1 = f'stty -F {self.port} {self.baud} cs8 -cstopb -parenb -echo'
		try:
			subprocess.run(command1, shell=True, check=True)
		except subprocess.CalledProcessError as e:
			self.get_logger().error(f"Error while setting up serial port: {e}")

	def read_sonde_data(self):
		if self.ser.in_waiting > 0:
			line = self.ser.readline()
			try:
				line = line.decode('utf-8').rstrip()
			except UnicodeDecodeError:
				self.get_logger().warn("Failed to decode line")
				return
			if line.startswith("#DATA:"):
				formatted_line = line.replace('#DATA: ', '')
				data = formatted_line.split(',')
				formatted_data = "".join(f"{item:<{width}}" for item, width in zip(data, self.column_widths))

				print(formatted_data)
				msg = String()
				msg.data = formatted_data
				self.publisher_.publish(msg)

	def shutdown(self):
		self.ser.close()
		self.get_logger().info("Serial port closed")


def main(args=None):
	rclpy.init(args=args)
	node = SondeReader()

	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	finally:
		node.destroy_node()
		node.shutdown()


if __name__ == '__main__':
	main()
