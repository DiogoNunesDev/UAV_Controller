import matplotlib.pyplot as plt
import math
import numpy as np

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c 

def plot_path_with_target_and_quadrants(filename="best_individual_log.txt"):

    latitudes = []
    longitudes = []
    altitudes = []
    target_lat, target_lon = None, None
    
    with open(filename, "r") as file:
        lines = file.readlines()

        target_line = lines[1].strip().split(", ")
        target_lat = float(target_line[0].split(": ")[1])
        target_lon = float(target_line[1].split(": ")[1])

        for line in lines[3:]:  
            if line.strip():  
                parts = line.strip().split("\t")
                if len(parts) == 4:
                    _, lat, lon, alt = parts
                    latitudes.append(float(lat))
                    longitudes.append(float(lon))
                    altitudes.append(float(alt))

    start_lat, start_lon = latitudes[0], longitudes[0]
    end_lat, end_lon = latitudes[-1], longitudes[-1]
    dist_start_target = haversine(start_lat, start_lon, target_lat, target_lon)
    dist_start_end = haversine(start_lat, start_lon, end_lat, end_lon)
    dist_target_end = haversine(target_lat, target_lon, end_lat, end_lon)

    print(f"Distance from Start to Target: {dist_start_target:.2f} meters")
    print(f"Distance from Start to End: {dist_start_end:.2f} meters")
    print(f"Distance from Target to End: {dist_target_end:.2f} meters")


    steps = list(range(1, len(latitudes) + 1))

    plt.figure(figsize=(16, 8))

    # Subplot 1: Trajectory Path
    plt.subplot(1, 2, 1)
    plt.plot(longitudes, latitudes, marker='o', color="blue", label="Path")
    plt.scatter(start_lon, start_lat, color="green", s=100, label="Start Point")
    plt.scatter(target_lon, target_lat, color="purple", s=100, label="Specified Target Point")
    plt.scatter(end_lon, end_lat, color="red", s=100, label="End Point")

    min_lon, max_lon = min(longitudes), max(longitudes)
    min_lat, max_lat = min(latitudes), max(latitudes)
    mid_lon = (max_lon + min_lon) / 2
    mid_lat = (max_lat + min_lat) / 2

    plt.axhline(y=mid_lat, color='black', linestyle='--', linewidth=0.5)  
    plt.axvline(x=mid_lon, color='black', linestyle='--', linewidth=0.5)  

    plt.text(mid_lon, max_lat, 'North', ha='center', va='bottom', fontsize=12, color='black')
    plt.text(mid_lon, min_lat, 'South', ha='center', va='top', fontsize=12, color='black')
    plt.text(max_lon, mid_lat, 'East', ha='left', va='center', fontsize=12, color='black')
    plt.text(min_lon, mid_lat, 'West', ha='right', va='center', fontsize=12, color='black')

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Trajectory Path with Specified Target and Quadrants")
    plt.legend()
    plt.grid(True)

    # Subplot 2: Altitude across Steps
    plt.subplot(1, 2, 2)
    plt.plot(steps, altitudes, marker='o', color="purple")
    plt.xlabel("Step")
    plt.ylabel("Altitude (m)")
    plt.title("Altitude of Aircraft across Steps")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Run the function to plot the path and altitude graph
plot_path_with_target_and_quadrants("best_individual_log.txt")
