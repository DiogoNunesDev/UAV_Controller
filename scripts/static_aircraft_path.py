import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculates the great-circle distance between two points on the Earth (in meters).
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
    earth_radius = 6378137.0  # Radius of Earth in meters
    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)
    x = delta_lon * earth_radius * np.cos(np.radians(lat1))
    y = delta_lat * earth_radius
    return x, y

def divide_into_segments(file_path):
    """
    Divides the file into segments (episodes) based on "Target Latitude".
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
    Parses a segment to extract trajectory data and target information.
    """
    data = []
    target_lat, target_lon, target_alt = None, None, None

    for line in segment:
        line = line.strip()
        if not line or line.startswith("Best Fitness"):  
            continue
        if "Target Latitude" in line:
            target_line = line.split(',')
            target_lat = float(target_line[0].split(':')[1].strip())
            target_lon = float(target_line[1].split(':')[1].strip())
            target_alt = float(target_line[2].split(':')[1].replace('m', '').strip())
        elif line[0].isdigit():  # Ensures the line starts with a digit (valid step data)
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
        [0, -0.5]   # Close the triangle
    ]) * size

    rotation_matrix = np.array([
        [np.cos(np.radians(yaw)), -np.sin(np.radians(yaw))],
        [np.sin(np.radians(yaw)), np.cos(np.radians(yaw))]
    ])
    rotated_airplane = airplane @ rotation_matrix.T
    translated_airplane = rotated_airplane + np.array([x, y])
    airplane_patch = patches.Polygon(translated_airplane, closed=True, color='blue', alpha=0.8)
    ax.add_patch(airplane_patch)

def plot_trajectory(data, target, initial_lat, initial_lon, episode_num):
    """
    Plot the trajectory for a single episode.
    """
    fig, ax = plt.subplots(figsize=(15, 15))

    ax.set_xlim(-300, 300)
    ax.set_ylim(-300, 300)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_title(f"Episode {episode_num}: 2D Airplane Trajectory with Target")
    ax.set_xlabel("East/West (meters)")
    ax.set_ylabel("North/South (meters)")
    ax.grid(True)

    trajectory_x, trajectory_y = [], []

    for point in data:
        x, y = lat_lon_to_meters(initial_lat, initial_lon, point['latitude'], point['longitude'])
        trajectory_x.append(x)
        trajectory_y.append(y)
        draw_airplane(ax, x, y, point['heading'], size=3)

    ax.plot(trajectory_x, trajectory_y, linestyle='dashed', color='gray', alpha=0.7, label="Trajectory Path")

    target_x, target_y = lat_lon_to_meters(initial_lat, initial_lon, target[0], target[1])
    ax.plot(target_x, target_y, 'ro', markersize=10, label="Target")
    ax.text(target_x + 5, target_y + 5, "Target", color='red', fontsize=12)

    ax.legend()
    plt.show()

    # Plot altitude variation
    plt.figure(figsize=(10, 5))
    plt.plot([point['altitude'] for point in data], marker='o', color='orange', label="Altitude")
    plt.title(f"Episode {episode_num}: Altitude Variation")
    plt.xlabel("Step")
    plt.ylabel("Altitude (m)")
    plt.grid(True)
    plt.legend()
    plt.show()

def main(file_path):
    """
    Main function to process the file and plot trajectories for each episode.
    """
    segments = divide_into_segments(file_path)

    for episode_num, segment in enumerate(segments, start=0):
        data, target = parse_segment(segment)
        if data:
            initial_lat, initial_lon = data[0]['latitude'], data[0]['longitude']
            final_lat, final_lon = data[-1]['latitude'], data[-1]['longitude']  # Final step coordinates

            distance_to_target = haversine(final_lat, final_lon, target[0], target[1])
            print(f"Episode {episode_num}: Distance from final step to target = {distance_to_target:.2f} meters")

            plot_trajectory(data, target, initial_lat, initial_lon, episode_num)


# Main Execution
main("../txt_files/ppo_log.txt")
