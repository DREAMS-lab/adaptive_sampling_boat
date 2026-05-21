#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import serial
import time
import subprocess

class SondeReader(Node):
	def __init__(self):
		super().__init__('sonde_reader') #initializes the Node 'sonde_reader'
		self.publisher_=self.create_publisher(String, 'sonde_data',10)
		self.timer=self.create_timer(0.1, self.read_sonde_data) #0.1 refers to 10Hz
		self.setup_serial_port() #runs my command that sets the baud rate of the sonde
		self.ser= serial.Serial('/dev/ttyUSB2', baudrate=19200, timeout=1)
		self.ser.write(b"VER\r")  #This command verifies that the firmware of the sonde is working correctly
		time.sleep(0.5)
		self.ser.write(b"SCR -on\r") #this command tells the sonde data to start logging

		self.columns= ["DATA", "TIME", "VOID", "Tempe dec C", "pH units", "Depth m", "SpCond uS/cm", "HDO %Sat", "HDO mg/l", "Chl ug/l", "CDOM ppb", "Turb NTU"]
		self.column_widths =[10,9,9,12,10,8,14,9,9,9,10,9]

		header= "".join(f"{name:<{width}}" for name,width in zip(self.columns, self.column_widths))
		print(header)

	def setup_serial_port(self):  #This sets up baud rate of sonde
		command1= 'stty -F /dev/ttyUSB2 19200 cs8 -cstopb -parenb -echo'
		try:

			subprocess.run(command1, shell=True, check=True)
		except subprocess.CalledProcessError as e:
			self.get_logger().error(f"Error while setting up serial port: {e}")


	def read_sonde_data(self): #Controls data logging
		if self.ser.in_waiting > 0:
			line=self.ser.readline()
			try:
				line=line.decode('utf-8').rstrip()
			except UnicodeDecodeError:
				self.get_logger().warn("Failed to decode line")
				return
			if line.startswith("#DATA:"):
				formatted_line=line.replace('#DATA: ','')
				data= formatted_line.split(',')
				formatted_data= "".join(f"{item:<{width}}" for item, width in zip(data,self.column_widths))

				print(formatted_data)
				msg= String()
				msg.data=formatted_data
				self.publisher_.publish(msg)
	def shutdown(self):
		self.ser.close()
		self.get_logger().info("Serial port closed")
def main(args=None):
	rclpy.init(args=args)
	node= SondeReader()

	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		#node.get_logger().info("Shutting down...")
		pass

	finally:
		node.destroy_node()
		node.shutdown()
if __name__ == '__main__':
	main()
