#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
import serial
import time

def read_sonde_data():


# Open the serial port
    ser = serial.Serial('/dev/ttyUSB0', baudrate=19200, timeout=1)


    rospy.init_node('sonde_reader', anonymous=True)
    pub = rospy.Publisher('sonde_data', String, queue_size=10)
    rate = rospy.Rate(10)  # 10 Hz

    # Define column names and their widths
    columns = ["DATE", "TIME", "VOID", "Temp deg C", "pH units", "Depth m", "SpCond uS/cm", "HDO %Sat", "HDO mg/l", "Chl ug/l", "CDOM ppb", "Turb NTU"]
    column_widths = [10, 9, 9, 12, 10, 8, 14, 9, 9, 9, 10, 9]

    # Print column names
    header = "".join(f"{name:<{width}}" for name, width in zip(columns, column_widths))
    print(header)

    try:
        while not rospy.is_shutdown():
            # Check if there is data waiting in the serial buffer
            if ser.in_waiting > 0:
                line = ser.readline()
                try:
                    line = line.decode('utf-8').rstrip()
                except UnicodeDecodeError:
                    # Handle decoding error
                    rospy.logwarn("Failed to decode line")
                    continue

                # Filter and format the data line
                if line.startswith("#DATA:"):
                    formatted_line = line.replace('#DATA: ', '')
                    data = formatted_line.split(',')
                    
                    # Format data according to column widths
                    formatted_data = "".join(f"{item:<{width}}" for item, width in zip(data, column_widths))
                    print(formatted_data)
                    pub.publish(formatted_data)

            rate.sleep()

    except rospy.ROSInterruptException:
        pass
    finally:
        # Close the serial port
        ser.close()
        rospy.loginfo("Serial port closed.")

if __name__ == '__main__':
    read_sonde_data()
