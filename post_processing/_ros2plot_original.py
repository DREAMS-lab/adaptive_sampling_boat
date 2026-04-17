#!/usr/bin/env python3

import math
import gsw
import folium
from folium import Element
import os
import bisect
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import csv
import plotly.graph_objects as go
from datetime import datetime
from folium.plugins import HeatMap
import branca.colormap as cm
import statistics
from pathlib import Path

import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions

rclpy.init()


# Function to get a valid ROS2 bag title from user input
def get_rosbag_title():
    while True:
        title = input("Please enter title for the ROS2 bag: ").strip()
        if title and os.path.isdir(title):  # Ensure the title is not empty
            if os.path.isfile(os.path.join(title, 'metadata.yaml')):
                return title
            else:
                print("Error: Folder does not have 'metadata.yaml' file")
        else:
            print("Error: Invalid folder path.  Please try again.")
       
def add_sonar_depth():
    while True:
        try:
            return float(input("Enter a value to add to every sonar reading (in meters): "))
        except ValueError:
            print("Invalid input. Please enter a numeric value.")
sonar_offset= add_sonar_depth()

rosbag_title = get_rosbag_title()
# Topics to extract
topics_of_interest = [
    '/mavros/global_position/global',
    '/ping1d/data',
    '/sonde_data'
]

# Initialize storage
reader = SequentialReader()
storage_options = StorageOptions(uri=rosbag_title, storage_id='mcap')
converter_options = ConverterOptions('', '')
reader.open(storage_options, converter_options)

# Get all topic types
topic_types = reader.get_all_topics_and_types()
type_map = {topic.name: topic.type for topic in topic_types}

# Prepare data lists
gps_data = []
sonar_data = []
sonde_data_list = []
salinity_psu= []

# Read messages
while reader.has_next():
    topic, data, t = reader.read_next()
    timestamp = t / 1e9  # nanoseconds to seconds

    if topic not in topics_of_interest:
        continue

    msg_type_str = type_map[topic]
    msg_type = get_message(msg_type_str)
    msg = deserialize_message(data, msg_type)

    if topic == '/mavros/global_position/global':
        gps_data.append((timestamp, msg.latitude, msg.longitude))

    elif topic == '/ping1d/data':
        adjusted_range = msg.data + sonar_offset
        sonar_data.append((timestamp, adjusted_range))


    elif topic == '/sonde_data':
        sonde_values = [value.strip() for value in msg.data.split() if value]

        if len(sonde_values) == 12:
            sonde_data = {
                "timestamp": timestamp,
                "Date": sonde_values[0],
                "Time": sonde_values[1],
                "Temp deg C": sonde_values[3],
                "pH units": sonde_values[4],
                "Depth m": sonde_values[5],
                "SpCond uS/cm": sonde_values[6],
                "HDO sat": sonde_values[7],
                "HDO mg/L": sonde_values[8],
                "Chl ug/L": sonde_values[9],
                "CDOM ppb": sonde_values[10],
                "Turb NTU": sonde_values[11]
            }
            sonde_data_list.append(sonde_data)
        elif len(sonde_values) == 11:
            sonde_data = {
                "timestamp": timestamp,
                "Date": sonde_values[0],
                "Time": sonde_values[1],
                "Temp deg C": sonde_values[3],
                "pH units": sonde_values[4],
                "Depth m": sonde_values[5],
                "SpCond uS/cm": sonde_values[6],
                "HDO sat": sonde_values[7],
                "HDO mg/L": sonde_values[8],
                "Chl ug/L": sonde_values[9],
                "CDOM ppb": sonde_values[10]
            }
            sonde_data_list.append(sonde_data)

# Shutdown rclpy once done
rclpy.shutdown()

#Calculate Salinity
def pressure_from_depth(depth_m, con, temp, latitude_deg, longitude_deg):
    depth_m = float(depth_m)
    con = float(con)
    temp = float(temp)
    latitude_deg = float(latitude_deg)
    longitude_deg = float(longitude_deg)
    pressure= gsw.p_from_z(depth_m, latitude_deg)
    con_ms=con/1000
    SP = gsw.SP_from_C(con_ms, temp, pressure)
    SA = gsw.SA_from_SP(SP, pressure, longitude_deg, latitude_deg)
    
    if np.isnan(SP):
    	SP=0
    if np.isnan(SA):
    	SA=0
    return SP,SA


# Function to find the closest timestamp in a list of data
def find_closest(data_list, target_timestamp):
    timestamps = [data[0] for data in data_list]
    pos = bisect.bisect_left(timestamps, target_timestamp)
    if pos == 0:
        return data_list[0]
    if pos == len(data_list):
        return data_list[-1]
    before = data_list[pos - 1]
    after = data_list[pos]
    return before if target_timestamp - before[0] < after[0] - target_timestamp else after

# Extract the first Date and Time values from the first sonde data entry
if sonde_data_list:
    first_sonde = sonde_data_list[0]
    first_date = first_sonde['Date'].replace('/', '-')  # Replace slashes with dashes
    first_time = first_sonde['Time'].replace(':', '-')  # Replace colons with dashes
    title_info = f"{first_date}_{first_time}"
else:
    title_info = f"{rosbag_title}_Unknown_Sonde_Time"  # Fallback in case there's no sonde data


##############################################3
# --- Write Data to CSV ---
csv_filename = f'{title_info}_raw_data.csv'



# Open the CSV file for writing
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)

    # Write the header (column names)
    writer.writerow(['Time (UTC)', 'Latitude', 'Longitude', 'Depth (Sonar)', 
                     'Temperature (°C)', 'pH', 'Depth (m)', 'Conductivity (uS/cm)', 
                     'Dissolved Oxygen Saturation', 'Dissolved Oxygen Concentration (mg/L)', 
                     'Chlorophyll (ug/L)', 'CDOM (ppb)', 'Turbidity (NTU)', 'Salinity (PSU)'])

    # Write the data
    for sonde in sonde_data_list:
        # Find the closest GPS and sonar data points to the current sonde timestamp
        closest_gps = find_closest(gps_data, sonde['timestamp'])
        closest_sonar = find_closest(sonar_data, sonde['timestamp'])
        salinity=pressure_from_depth(sonde['Depth m'], sonde['SpCond uS/cm'], sonde['Temp deg C'], closest_gps[1], closest_gps[2])
        salinity_psu.append(float(salinity[0]))


   
        
        # Prepare data for this row
        row = [
            sonde['timestamp'],  # Time
            closest_gps[1],      # Latitude
            closest_gps[2],      # Longitude
            closest_sonar[1],    # Depth from Sonar
            sonde['Temp deg C'], # Temperature
            sonde['pH units'],   # pH
            sonde['Depth m'],    # Depth (from Sonde)
            sonde['SpCond uS/cm'], # Conductivity
            sonde['HDO sat'],    # Dissolved Oxygen Saturation
            sonde['HDO mg/L'],   # Dissolved Oxygen Concentration
            sonde['Chl ug/L'],   # Chlorophyll
            sonde['CDOM ppb'],   # CDOM
            sonde.get('Turb NTU', ''),  # Turbidity (leave blank if not available)
            salinity[0]
        ]
        writer.writerow(row)

# Inform the user the CSV was created
print(f"CSV file saved as {csv_filename}")


################################################################

# Create a folium map centered at the mean of the GPS coordinates
if gps_data:
    mean_lat = sum(lat for _, lat, _ in gps_data) / len(gps_data)
    mean_lon = sum(lon for _, _, lon in gps_data) / len(gps_data)
    m = folium.Map(location=[mean_lat, mean_lon], zoom_start=15, control_scale=True)

    # Move zoom controls to bottom right
    zoom_position = Element("""
    <script>
        var map = document.getElementsByClassName('leaflet-container')[0]._leaflet_map;
        map.zoomControl.setPosition('bottomright');
    </script>
    """)
    m.get_root().html.add_child(zoom_position)

    # Define colormaps for each parameter
    colormaps = {
        'Temp deg C': cm.linear.YlOrRd_09.scale(min(float(sonde['Temp deg C']) for sonde in sonde_data_list), max(float(sonde['Temp deg C']) for sonde in sonde_data_list)),
        'pH units': cm.linear.PuBu_09.scale(min(float(sonde['pH units']) for sonde in sonde_data_list), max(float(sonde['pH units']) for sonde in sonde_data_list)),
        'SpCond uS/cm': cm.linear.Oranges_09.scale(min(float(sonde['SpCond uS/cm']) for sonde in sonde_data_list), max(float(sonde['SpCond uS/cm']) for sonde in sonde_data_list)),
        'HDO mg/L': cm.linear.Blues_09.scale(min(float(sonde['HDO mg/L']) for sonde in sonde_data_list), max(float(sonde['HDO mg/L']) for sonde in sonde_data_list)),
        'Chl ug/L': cm.linear.Greens_09.scale(min(float(sonde['Chl ug/L']) for sonde in sonde_data_list), max(float(sonde['Chl ug/L']) for sonde in sonde_data_list)),
        'CDOM ppb': cm.linear.Reds_09.scale(min(float(sonde['CDOM ppb']) for sonde in sonde_data_list), max(float(sonde['CDOM ppb']) for sonde in sonde_data_list)),
        'Sonar Depth': cm.linear.Purples_09.scale(min(float(sonar[1]) for sonar in sonar_data), max(float(sonar[1]) for sonar in sonar_data)),
        'Salinity (PSU)': cm.linear.BuGn_09.scale(min(salinity_psu),max(salinity_psu))
    }

    if any('Turb NTU' in sonde for sonde in sonde_data_list):
        colormaps['Turb NTU'] = cm.linear.Greys_09.scale(
            min(float(sonde['Turb NTU']) for sonde in sonde_data_list if 'Turb NTU' in sonde),
            max(float(sonde['Turb NTU']) for sonde in sonde_data_list if 'Turb NTU' in sonde)
        )

    # Create layers for each parameter
    layers = {param: folium.FeatureGroup(name=param).add_to(m) for param in colormaps.keys()}
    param_stats = {param: [] for param in colormaps.keys()}
    sidebar_stats = ""

    for sonde in sonde_data_list:
        closest_gps = find_closest(gps_data, sonde['timestamp'])
        closest_sonar = find_closest(sonar_data, sonde['timestamp'])
        lat, lon = closest_gps[1], closest_gps[2]
        depth = float(closest_sonar[1])
        salinity=pressure_from_depth(sonde['Depth m'], sonde['SpCond uS/cm'], sonde['Temp deg C'], closest_gps[1], closest_gps[2])
        
        for param, colormap in colormaps.items():
            if param in sonde:
                value = float(sonde[param])
                param_stats[param].append((value, lat, lon))
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=2,
                    color=colormap(value),
                    fill=True,
                    fill_color=colormap(value),
                    fill_opacity=0.7,
                    popup=f"{param}: {value}"
                ).add_to(layers[param])

        # Add sonar depth
        param_stats['Sonar Depth'].append((depth, lat, lon))
        folium.CircleMarker(
            location=[lat, lon],
            radius=2,
            color=colormaps['Sonar Depth'](depth),
            fill=True,
            fill_color=colormaps['Sonar Depth'](depth),
            fill_opacity=0.7,
            popup=f"Sonar Depth: {depth} meters"
        ).add_to(layers['Sonar Depth'])

        # Add salinity
        param_stats['Salinity (PSU)'].append((salinity[0], lat, lon))
        folium.CircleMarker(
            location=[lat, lon],
            radius=2,
            color=colormaps['Salinity (PSU)'](salinity[0]),
            fill=True,
            fill_color=colormaps['Salinity (PSU)'](salinity[0]),
            fill_opacity=0.7,
            popup=f"Salinity (PSU): {salinity[0]}"
        ).add_to(layers['Salinity (PSU)'])
    # Add stats + sidebar text
    for param, values in param_stats.items():
        if values:
            val_list, lat_list, lon_list = zip(*values)
            mean_val = np.mean(val_list)
            min_val = min(val_list)
            max_val = max(val_list)

            try:
                mode_val = statistics.mode(val_list)
            except statistics.StatisticsError:
                mode_val = val_list[0] if val_list else 'N/A'

            mean_idx = min(range(len(val_list)), key=lambda i: abs(val_list[i] - mean_val))
            min_idx = val_list.index(min_val)
            max_idx = val_list.index(max_val)

            mean_loc = (lat_list[mean_idx], lon_list[mean_idx])
            min_loc = (lat_list[min_idx], lon_list[min_idx])
            max_loc = (lat_list[max_idx], lon_list[max_idx])

            # Add markers on map
            folium.Marker(
                location=mean_loc,
                icon=folium.Icon(color='blue', icon='info-sign'),
                popup=f"{param} Mean: {mean_val:.2f}"
            ).add_to(layers[param])
            folium.Marker(
                location=min_loc,
                icon=folium.Icon(color='green', icon='info-sign'),
                popup=f"{param} Min: {min_val:.2f}"
            ).add_to(layers[param])
            folium.Marker(
                location=max_loc,
                icon=folium.Icon(color='red', icon='info-sign'),
                popup=f"{param} Max: {max_val:.2f}"
            ).add_to(layers[param])

            # Add to sidebar
            sidebar_stats += f"""
                <div style='margin-bottom:10px'>
                    <b>{param}</b><br>
                    <span style='color:blue'>Mean: {mean_val:.2f}</span><br>
                    <span style='color:green'>Min: {min_val:.2f}</span><br>
                    <span style='color:red'>Max: {max_val:.2f}</span>
                </div>
            """

    # Add sidebar as HTML
    stats_html = Element(f"""
    <div style="
        position: fixed;
        top: 55%;
        left: 8px;
        transform: translateY(-50%);
        background-color: white;
        padding: 9px;
        border: 2px solid gray;
        border-radius: 10px;
        max-height: 80%;
        overflow-y: auto;
        z-index: 9999;
        font-size: 13px;
    ">
        <h4 style='margin:0 0 10px 0'>Stats</h4>
        {sidebar_stats}
    </div>
    """)
    m.get_root().html.add_child(stats_html)

    # Add colormaps and controls
    for param, colormap in colormaps.items():
        colormap.caption = param
        colormap.add_to(m)
    folium.LayerControl().add_to(m)

    # Save map
    m.save(f'{title_info}_lake_depth_sonde_map.html')







##################

# --- Figure 2: Sonde Characteristics vs GPS Coordinates (3x3 Grid) ---
# Prepare lists for sonde data characteristics
temperature = []
ph = []
depth = []
conductivity = []
dissolved_oxygen_sat = []
dissolved_oxygen_con = []
chlorophyll = []
cdom = []
turbidity = []
salt= []

# List for corresponding GPS latitudes and longitudes
gps_latitudes = []
gps_longitudes = []

# Only match GPS data with sonde data that share the same timestamp
for sonde in sonde_data_list:
    # Find the closest GPS point for each sonde point
    closest_gps = find_closest(gps_data, sonde['timestamp'])
    gps_latitudes.append(closest_gps[1])  # Latitude
    gps_longitudes.append(closest_gps[2])  # Longitude
    sal=pressure_from_depth(sonde['Depth m'], sonde['SpCond uS/cm'], sonde['Temp deg C'], closest_gps[1], closest_gps[2])
    salt.append(float(sal[0]))

    # Add corresponding sonde data
    temperature.append(float(sonde['Temp deg C']))
    ph.append(float(sonde['pH units']))
    depth.append(float(sonde['Depth m']))
    conductivity.append(float(sonde['SpCond uS/cm']))
    dissolved_oxygen_sat.append(float(sonde['HDO sat']))
    dissolved_oxygen_con.append(float(sonde['HDO mg/L']))
    chlorophyll.append(float(sonde['Chl ug/L']))
    cdom.append(float(sonde['CDOM ppb']))
    if len(sonde_values)==12:
    	turbidity.append(float(sonde['Turb NTU']))

# Create a 3-row, 4-column grid of subplots
fig2, axes = plt.subplots(3, 4, figsize=(12, 10))  # Slightly wider for 4 plots

# Sonde data columns and values including salt
sonde_columns = [
    'Temperature (°C)', 'pH', 'Depth (m)', 'Conductivity (uS/cm)',
    'Dissolved Oxygen Saturation (sat)', 'Dissolved Oxygen Concentration (mg/L)',
    'Chlorophyll (ug/L)', 'CDOM (ppb)', 'Turbidity (NTU)', 'Salinity (PSU)'
]

data = [
    temperature, ph, depth, conductivity,
    dissolved_oxygen_sat, dissolved_oxygen_con,
    chlorophyll, cdom, turbidity, salt
]

# If only 11 fields were expected, remove turbidity to match old behavior
if len(sonde_values) == 11:
    sonde_columns = sonde_columns[:-2] + ['Salt']  # Remove 'Turbidity', keep 'Salt'
    data = data[:-2] + [salt]

# Flatten axes for easier indexing
axes_flat = axes.flatten()

# Plot each characteristic
for i, (ax, characteristic, characteristic_data) in enumerate(zip(axes_flat, sonde_columns, data)):
    scatter = ax.scatter(gps_longitudes, gps_latitudes, c=characteristic_data, cmap='viridis', s=1)
    ax.set_title(characteristic, fontsize=6)
    ax.set_xlabel('Longitude', fontsize=6)
    ax.set_ylabel('Latitude', fontsize=6)
    ax.grid(True)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(axis='both', which='minor', labelsize=6)
    ax.set_xlim([min(gps_longitudes) - 0.0001, max(gps_longitudes) + 0.0001])
    ax.set_ylim([min(gps_latitudes) - 0.0001, max(gps_latitudes) + 0.0001])
    ax.ticklabel_format(style='sci', scilimits=(0, 0), axis='both', useMathText=True)
    ax.xaxis.get_offset_text().set_x(1.2) 
    ax.xaxis.get_offset_text().set_fontsize(5)
    ax.yaxis.get_offset_text().set_fontsize(5)

    cbar = fig2.colorbar(scatter, ax=ax)
    cbar.ax.tick_params(labelsize=6)

# Turn off any unused axes (only needed if sonde_values < 12)
for i in range(len(data), len(axes_flat)):
    axes_flat[i].axis('off')

# Improve spacing and layout
fig2.tight_layout()

# Save figure
figure_name = f'{title_info}_Sonde_Water_Quality_vs_GPS_Coordinates.png'
fig2.savefig(figure_name, format='png')
plt.show()




"""
# Create a 3x3 grid of plots for sonde characteristics vs GPS coordinates
fig2, axes = plt.subplots(3, 3, figsize=(10, 10))  # Increased size for better visibility

if len(sonde_values)==11:
    for i in range(3):
        for j in range(3):
            if i == 2 and j > 1:  # Skip the last two subplots in the third row (making it 2 columns)
                axes[i, j].axis('off')


# List of sonde characteristics
sonde_columns = ['Temperature (°C)', 'pH', 'Depth (m)', 'Conductivity (uS/cm)', 'Dissolved Oxygen Saturation (sat)', 'Dissolved Oxygen Concentration (mg/L)','Chlorophyll (ug/L)', 'CDOM (ppb)', 'Turbidity (NTU)']

# Data corresponding to each characteristic
data = [temperature, ph, depth, conductivity, dissolved_oxygen_sat, dissolved_oxygen_con, chlorophyll, cdom, turbidity]

if len(sonde_values)==11:
	sonde_columns=sonde_columns[:-1]
	data=data[:-1]

# Plot each characteristic
for i, (ax, characteristic, characteristic_data) in enumerate(zip(axes.ravel(), sonde_columns, data)):
    scatter = ax.scatter(gps_longitudes, gps_latitudes, c=characteristic_data, cmap='viridis', s=1)
    ax.set_title(characteristic, fontsize=6)
    ax.set_xlabel('Longitude', fontsize=6)
    ax.set_ylabel('Latitude', fontsize=6)
    ax.grid(True)
    ax.tick_params(axis='both', which='major', labelsize=8)  # Major tick font size
    ax.tick_params(axis='both', which='minor', labelsize=6)  # Minor tick font size
    
    # Set axis limits based on the data range for zoom-in effect
    ax.set_xlim([min(gps_longitudes) - 0.0001, max(gps_longitudes) + 0.0001])  # Adjust the limits for better zoom
    ax.set_ylim([min(gps_latitudes) - 0.0001, max(gps_latitudes) + 0.0001])  # Adjust the limits for better zoom
    
    ax.ticklabel_format(style='sci', scilimits=(0,0), axis='both', useMathText=True)
    ax.xaxis.get_offset_text().set_fontsize(5)  # Change the font size here
    ax.yaxis.get_offset_text().set_fontsize(5)    
    
    # Add colorbar to each subplot
    cbar = fig2.colorbar(scatter, ax=ax)
    cbar.ax.tick_params(labelsize=6)  # Change the font size of the colorbar ticks

# Adjust layout to ensure everything fits without overlap
fig2.tight_layout()

# Save figure as a PNG with a dynamic title including the first Date and Time
figure_name = f'{title_info}_Sonde_Water_Quality_vs_GPS_Coordinates.png'
fig2.savefig(figure_name, format='png')

plt.show()

"""
# --- Figure 3: Sonar Depth vs GPS Coordinates ---
# Extract the sonar depth data (matching timestamp to GPS coordinates)
sonar_depth = []
sonar_latitudes = []
sonar_longitudes = []

for sonar in sonar_data:
    closest_gps = find_closest(gps_data, sonar[0])  # Matching timestamp
    sonar_depth.append(sonar[1])
    sonar_latitudes.append(closest_gps[1])  # Latitude
    sonar_longitudes.append(closest_gps[2])  # Longitude

# Plot sonar depth vs GPS coordinates
fig3 = plt.figure(figsize=(10, 8))  # Increased figure size for better visibility
scatter = plt.scatter(sonar_longitudes, sonar_latitudes, c=sonar_depth, cmap='plasma', s=1)
plt.colorbar(scatter, label='Sonar Depth (m)')
plt.xlabel('Longitude', fontsize=10)
plt.ylabel('Latitude', fontsize=10)
plt.title(f'Sonar_Depth_vs_GPS_Coordinates', fontsize=12)

# Set axis limits based on the data range for zoom-in effect
plt.xlim([min(sonar_longitudes) - 0.0001, max(sonar_longitudes) + 0.0001])  # Adjust the limits for better zoom
plt.ylim([min(sonar_latitudes) - 0.0001, max(sonar_latitudes) + 0.0001])  # Adjust the limits for better zoom

plt.grid(True)

figure_name2 = f'{title_info}_Depth_vs_GPS_Coordinates.png'
fig3.savefig(figure_name2, format='png')
plt.show()


###########
# Sonar Data Saved in Separate CSV, since sonar usually has higher count than sonde data
csv_filename = f'{title_info}_sonar_raw_data.csv'



# Open the CSV file for writing
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)

    # Write the header (column names)
    writer.writerow(['Time (UTC)', 'Latitude', 'Longitude', 'Depth (Sonar)'])

    # Write the data
    for sonar in sonar_data:
        closest_gps = find_closest(gps_data, sonar[0])  # Matching timestamp
        sonar_depth.append(sonar[1])
        sonar_latitudes.append(closest_gps[1])  # Latitude
        sonar_longitudes.append(closest_gps[2])  # Longitude 
       
        # Prepare data for this row
        row = [
            sonar[0],  # Time
            closest_gps[1],      # Latitude
            closest_gps[2],      # Longitude
            sonar[1]             # Depth from Sonar
        ]
        writer.writerow(row)

# Inform the user the CSV was created
print(f"CSV file saved as {csv_filename}")



##############################################################
"""

# --- Figure 4: 3D Contour Plot of Sonar Depth vs GPS Coordinates ---


import os
import matplotlib.pyplot as plt

# Example data
x = [1, 2, 3, 4]
y = [10, 20, 25, 30]

# Create plot
plt.figure()
plt.plot(x, y)
plt.title("Example Plot")

# --- Folder and filename setup ---
output_folder = "output_images"  # relative or absolute path
filename = "my_plot.png"
filepath = os.path.join(output_folder, filename)

# Create folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Save the figure
plt.savefig(filepath, dpi=300, bbox_inches='tight')

print(f"Plot saved to {filepath}")


"""
