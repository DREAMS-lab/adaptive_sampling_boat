#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
import serial
import time
import csv

def read_sonde_data():
    # Open the serial port
    ser = serial.Serial('/dev/ttyUSB2', baudrate=19200, timeout=1)

    # Initialize ROS node
    rospy.init_node('sonde_reader', anonymous=True)
    pub = rospy.Publisher('sonde_data', String, queue_size=10)
    rate = rospy.Rate(10)  # 10 Hz

    # Define column names and their widths
    columns = ["DATE", "TIME", "VOID", "Temp deg C", "pH units", "Depth m", "SpCond uS/cm", "HDO %Sat", "HDO mg/l", "Chl ug/l", "CDOM ppb", "Turb NTU"]
    column_widths = [10, 9, 9, 12, 10, 8, 14, 9, 9, 9, 10, 9]

    # Print column names (just for display)
    header = "".join(f"{name:<{width}}" for name, width in zip(columns, column_widths))
    print(header)

    # Open a CSV file in write mode
    csv_filename = 'sonde_data_log.csv'

    # Write the header to the CSV file if it doesn't already exist
    try:
        with open(csv_filename, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(columns)  # Write header
    except Exception as e:
        rospy.logwarn(f"Error opening CSV file for writing: {e}")
        return

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

                    # Ensure the data has the correct number of elements
                    if len(data) == len(columns) - 1:  # One less because of "VOID"
                        # Get the current date and time for logging
                        current_time = time.localtime()
                        date_str = time.strftime('%Y-%m-%d', current_time)
                        time_str = time.strftime('%H:%M:%S', current_time)

                        # Prepend date and time to the data
                        data = [date_str, time_str, ''] + data  # 'VOID' field left empty

                        # Format data according to column widths for display
                        formatted_data = "".join(f"{item:<{width}}" for item, width in zip(data, column_widths))
                        print(formatted_data)
                        pub.publish(formatted_data)

                        # Write data to CSV file
                        try:
                            with open(csv_filename, mode='a', newline='') as csvfile:
                                writer = csv.writer(csvfile)
                                writer.writerow(data)  # Write row to CSV
                        except Exception as e:
                            rospy.logwarn(f"Error writing to CSV file: {e}")
                            continue

            rate.sleep()

    except rospy.ROSInterruptException:
        pass
    finally:
        # Close the serial port
        ser.close()
        rospy.loginfo("Serial port closed.")

if __name__ == '__main__':
    read_sonde_data()

