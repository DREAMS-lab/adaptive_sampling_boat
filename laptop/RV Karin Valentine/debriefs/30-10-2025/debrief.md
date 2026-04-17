# DEBRIEFING FOR OCTOBER 25TH,2025

> Objectives:Testing out two boats to perform simultaneous mission

### Items Checklist
- Boat 1
- Boat 2
- Laptop
-Base Telemetry Radio
- Sonde 
- Sonde Bucket
- Cart
- RC Transmitter
- Tent & chairs
- Duct tape
- AC/DC converter for Ubiquiti Bullet
- Three batteries
- Base RTK GPS

### Field Launch Checklist - 
- Make sure Motors operate
- Plug in the Telemetry Radio into base station laptop
- Turn on RC Transmitter
- Plug in the Batteries into the boat & and turn on the Boat (both the FMU and Thruster switches)
- Open QGroundControl and wait for boat’s data to connect
- Go to Manual Mode and Arm Vehicle
- Use the RC Transmitter to slightly move the thrusters
- Ensure Sonde, Sonar, and GPS data is being collated in the boat’s computer
- SSH to the boat’s computer
- Source the terminal
- First, ros2 run the sonar port
- If the terminal opens and fails, that means the USBs on the boat need to be unplugged and plugged back in
- If the terminal stays opens, that means data is coming through
- In second terminal, source it, and execute the sonde script
- Finally, in another terminal, ros2 launch the MAVROS node
- Ensure Battery voltages are around 13.3-13.7V

### Pre-Flight Information
- Voltages:
- Boat Battery: 13.47V
- Ubiquiti Bullet Voltage Supply: 13.54V (1 battery) 

### Flight Information

- Start of test: 1:47PM
- End of test: 3:15PM

> Participants: Rodney, Kanav, Bharath, Carlos
> Carlos was the photographer. Kanav helped with the set-up at the lake (along with everyone else) and watching the boat during its mission. Bharath helped with pre-flight checks. Overall, Rodney headed the operations, and checks during the whole process
What worked: sensor pinging, data gathering, better turns and PID 

### Issues:
RTK started well, but GPS on the boat eventually started registering as missing (towards the halfway point)
Resolution: Unplugged it. The boat’s GPS came back on. I think we left a box unchecked beforehand for RTK. 

### Post Flight Information

It wasn’t able to complete the mission (200m x 180m), due to battery depletion. While the start of the mission recorded a battery percentage of 78%, this went down to 15-25%. Nonetheless, this demonstrated the need for the second boat. 

## Mission from the FLight Controller

## Sone Reading Plots

## Battery data

## Sonar data 

