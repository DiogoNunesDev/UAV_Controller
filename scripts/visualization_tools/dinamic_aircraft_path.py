import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math
import time

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on the Earth (in meters).
    """
    R = 6371000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def lat_lon_to_meters(lat1, lon1, lat2, lon2):
    """
    Converts latitude and longitude differences to meters.
    """
    earth_radius = 6378137.0
    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)
    x = delta_lon * earth_radius * np.cos(np.radians(lat1))
    y = delta_lat * earth_radius
    return x, y

def divide_into_segments(file_path):
    """
    Divide the file into segments (episodes) based on "Target Latitude".
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    segments = []
    current_segment = []
    for line in lines:
        if "Target Latitude" in line:
            if current_segment:
                segments.append(current_segment)
            current_segment = [line.strip()]
        elif line.strip():
            current_segment.append(line.strip())

    if current_segment:
        segments.append(current_segment)

    return segments

def parse_segment(segment):
    """
    Parse a segment to extract trajectory data and target information.
    """
    data = []
    target_lat, target_lon, target_alt = None, None, None

    for line in segment:
        line = line.strip()
        if "Target Latitude" in line:
            target_line = line.split(',')
            target_lat = float(target_line[0].split(':')[1].strip())
            target_lon = float(target_line[1].split(':')[1].strip())
            target_alt = float(target_line[2].split(':')[1].replace('m', '').strip())
        elif line[0].isdigit():
            parts = line.split('\t')
            if len(parts) >= 5:
                data.append({
                    'step': int(parts[0]),
                    'latitude': float(parts[1]),
                    'longitude': float(parts[2]),
                    'altitude': float(parts[3]),
                    'heading': float(parts[4])
                })
    return data, (target_lat, target_lon, target_alt)

def draw_airplane(ax, x, y, yaw, size=1.0):
    """
    Draws an airplane symbol at (x, y) with orientation given by yaw (in degrees).
    """
    airplane = np.array([
        [0, -0.5],  # Tail
        [1, 0],     # Nose
        [0, 0.5],   # Tail
        [0, -0.5]
    ]) * size

    rotation_matrix = np.array([
        [np.cos(np.radians(yaw)), -np.sin(np.radians(yaw))],
        [np.sin(np.radians(yaw)), np.cos(np.radians(yaw))]
    ])
    rotated_airplane = airplane @ rotation_matrix.T
    translated_airplane = rotated_airplane + np.array([x, y])
    airplane_patch = patches.Polygon(translated_airplane, closed=True, color='blue', alpha=0.8)
    ax.add_patch(airplane_patch)

def plot_moving_trajectory(data, target, initial_lat, initial_lon, episode_num):
    """
    Animates the trajectory for a single episode with a panning map that follows the aircraft.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    window_size = 300  # Fixed viewing window in meters
    
    trajectory_x, trajectory_y = [], []
    target_x, target_y = lat_lon_to_meters(initial_lat, initial_lon, target[0], target[1])
    
    for i, point in enumerate(data):
        x, y = lat_lon_to_meters(initial_lat, initial_lon, point['latitude'], point['longitude'])
        trajectory_x.append(x)
        trajectory_y.append(y)
        
        # Clear and redraw
        ax.clear()
        ax.set_xlim(x - window_size / 2, x + window_size / 2)
        ax.set_ylim(y - window_size / 2, y + window_size / 2)
        ax.grid(True)

        # Trajectory
        ax.plot(trajectory_x, trajectory_y, linestyle='dashed', color='gray', alpha=0.7)
        draw_airplane(ax, x, y, point['heading'], size=3)

        tx = min(max(target_x, x - window_size / 2), x + window_size / 2)
        ty = min(max(target_y, y - window_size / 2), y + window_size / 2)
        ax.plot(tx, ty, 'ro', markersize=10, label="Target")
        ax.text(tx + 5, ty + 5, "Target", color='red', fontsize=12)

        # Annotations
        ax.text(x - window_size / 2 + 20, y + window_size / 2 - 20, f"Step: {point['step']}", fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
        ax.text(x - window_size / 2 + 20, y + window_size / 2 - 35, f"Altitude: {point['altitude']}", fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
        #ax.text(x - window_size / 2 + 20, y + window_size / 2 - 50, f"Heading: {point['heading']}°", fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
        
        plt.pause(0.05)  # Animation speed
    
    final_lat, final_lon = data[-1]['latitude'], data[-1]['longitude']
    distance_to_target = haversine(final_lat, final_lon, target[0], target[1])
    print(f"Episode {episode_num}: Distance from final step to target = {distance_to_target:.2f} meters")
    plt.show()

def main(file_path):
    """
    Main function to process the file and animate trajectories for each episode.
    """
    segments = divide_into_segments(file_path)

    for episode_num, segment in enumerate(segments, start=1):
        data, target = parse_segment(segment)
        if data:
            initial_lat, initial_lon = data[0]['latitude'], data[0]['longitude']
            plot_moving_trajectory(data, target, initial_lat, initial_lon, episode_num)



if __name__ == "__main__":
    main("../../txt_files/ppo_log.txt")
