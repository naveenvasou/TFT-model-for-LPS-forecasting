import pandas as pd
import numpy as np

def process_tracks(file_path):
    """
    Reads a track file (either Lows or MDs), parses the content, and extracts relevant data.
    
    Parameters:
        file_path (str): Path to the track text file
    
    Returns:
        data (list): A list of track point data entries
    """
    data = []
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    current_genesis_date = None
    
    for line in lines:
        parts = line.split()
        
        # Identify genesis point for a new track
        if parts[0] == "start":
            num_points = int(parts[1])  # Not used, but describes number of points in track
            genesis_year = int(parts[2])
            genesis_month = int(parts[3])
            genesis_day = int(parts[4])
            genesis_hour = int(parts[5])
            current_genesis_date = f"{genesis_year}-{genesis_month:02d}-{genesis_day:02d} {genesis_hour:02d}:00:00"
        else:
            # Extract data for each hourly point of the track
            x_grid = int(parts[0])
            y_grid = int(parts[1])
            longitude = float(parts[2])
            latitude = float(parts[3])
            min_stream_function = float(parts[4])
            min_sea_level_pressure = float(parts[5])  # This will be our 'mslp'
            pressure_drop = float(parts[6])
            max_surface_wind = float(parts[7])
            avg_relative_humidity = float(parts[8])
            max_surface_geopotential = float(parts[9])
            land_sea_ratio = float(parts[10])         # This will be our 'ls_ratio'
            acepsl = float(parts[11])
            ace = float(parts[12])
            pdi = float(parts[13])
            ike = float(parts[14])
            year = int(parts[15])
            month = int(parts[16])
            day = int(parts[17])
            hour = int(parts[18])
            
            date_time = f"{year}-{month:02d}-{day:02d} {hour:02d}:00:00"
            data.append([
                current_genesis_date, date_time,
                latitude, longitude,
                pressure_drop, land_sea_ratio
            ])
    
    return data


# Process Lows and MDs data files
Lows_data = process_tracks('raw/Lows_79_19.txt')
MDs_data = process_tracks('raw/MDs_79_19.txt')

# Create DataFrame for Lows
df_Lows = pd.DataFrame(Lows_data, columns=["Genesis_Date", "DateTime", "Latitude", "Longitude", "mslp", "ls_ratio"])

# Assign unique ID per Genesis Date
unique_genesis_dates = pd.unique(df_Lows["Genesis_Date"])
genesis_id_map = {date: i+1 for i, date in enumerate(unique_genesis_dates)}
df_Lows["id"] = [genesis_id_map[date] for date in df_Lows["Genesis_Date"]]

# Save Lows data to CSV
df_Lows.to_csv("processed/Lows_track.csv", index=False)


# Create DataFrame for MDs
df_MDs = pd.DataFrame(MDs_data, columns=["Genesis_Date", "DateTime", "Latitude", "Longitude", "mslp", "ls_ratio"])

# Assign unique ID per Genesis Date
unique_genesis_dates = pd.unique(df_MDs["Genesis_Date"])
genesis_id_map = {date: i+1 for i, date in enumerate(unique_genesis_dates)}
df_MDs["id"] = [genesis_id_map[date] for date in df_MDs["Genesis_Date"]]

# Save MDs data to CSV
df_MDs.to_csv("processed/MDs_track.csv", index=False)


