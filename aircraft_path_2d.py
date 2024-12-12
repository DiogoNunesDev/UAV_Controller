import matplotlib.pyplot as plt
import math

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c

def divide_into_segments(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    segments = []
    current_segment = []
    for line in lines:
        if "Target Latitude" in line:
            if len(current_segment) > 0:
                segments.append(current_segment) 
            current_segment = [line.strip()]
        elif "Step" in line:
            current_segment.append(line.strip())
        elif "Fitness" not in line:
            current_segment.append(line.strip())

    if len(current_segment) > 0:
        segments.append(current_segment)
    
    return segments

def plot_path_with_target_and_quadrants(filename="best_individual_log.txt"):
    latitudes = []
    longitudes = []
    altitudes = []
    target_latitudes = []
    target_longitudes = []
    target_altitudes = []

    segments = divide_into_segments(filename)
    
    for segment in segments:
        latitudes.clear()
        longitudes.clear()
        altitudes.clear()
        target_latitudes.clear()
        target_longitudes.clear()
        target_altitudes.clear()

        for line in segment:
            if line.startswith("Target Latitude"): 
                target_line = line.strip().split(", ")
                if len(target_line) == 3:
                    target_lat = float(target_line[0].split(": ")[1])
                    target_lon = float(target_line[1].split(": ")[1])
                    target_alt = float(target_line[2].split(": ")[1].replace("m", ""))
                    target_latitudes.append(target_lat)
                    target_longitudes.append(target_lon)
                    target_altitudes.append(target_alt)
            elif line.startswith("Step"): 
                continue 
            elif line.strip(): 
                parts = line.strip().split("\t")
                if len(parts) == 5:
                    _, lat, lon, alt, heading = parts
                    latitudes.append(float(lat))
                    longitudes.append(float(lon))
                    altitudes.append(float(alt))

        if len(latitudes) > 0:
            plot_segment(latitudes, longitudes, altitudes, target_latitudes, target_longitudes)

def plot_segment(latitudes, longitudes, altitudes, target_latitudes, target_longitudes):
    x_positions = [0] 
    y_positions = [0] 
    headings = []  

    for i in range(1, len(latitudes)):
        prev_lat, prev_lon = latitudes[i-1], longitudes[i-1]
        curr_lat, curr_lon = latitudes[i], longitudes[i]
        
        prev_lat_rad = math.radians(prev_lat)
        prev_lon_rad = math.radians(prev_lon)
        curr_lat_rad = math.radians(curr_lat)
        curr_lon_rad = math.radians(curr_lon)
        
        delta_lon = curr_lon_rad - prev_lon_rad
        
        x = math.sin(delta_lon) * math.cos(curr_lat_rad)
        y = math.cos(prev_lat_rad) * math.sin(curr_lat_rad) - math.sin(prev_lat_rad) * math.cos(curr_lat_rad) * math.cos(delta_lon)
        bearing = math.degrees(math.atan2(x, y))
        
        if bearing < 0:
            bearing += 360
        
        headings.append(bearing)

        dist = haversine(prev_lat, prev_lon, curr_lat, curr_lon)

        x_positions.append(x_positions[-1] + dist * math.sin(math.radians(bearing)))
        y_positions.append(y_positions[-1] + dist * math.cos(math.radians(bearing)))

    plt.figure(figsize=(10, 10))
    
    # Draw arrows instead of points
    for i in range(1, len(x_positions)):
        dx = x_positions[i] - x_positions[i-1]
        dy = y_positions[i] - y_positions[i-1]
        plt.arrow(x_positions[i-1], y_positions[i-1], dx, dy, 
                  head_width=3, head_length=5, fc='r', ec='r')

    plt.scatter(0, 0, color="green", s=100, label="Start Point")
    
    if len(target_latitudes) > 0:
        target_dist = haversine(latitudes[0], longitudes[0], target_latitudes[0], target_longitudes[0])
        target_heading = math.degrees(math.atan2(target_longitudes[0] - longitudes[0], target_latitudes[0] - latitudes[0]))
        target_x = target_dist * math.sin(math.radians(target_heading))
        target_y = target_dist * math.cos(math.radians(target_heading))
        plt.scatter(target_x, target_y, color="purple", s=100, label="Target Point")
    
    plt.xlim(-300, 300)
    plt.ylim(-300, 300)
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title("Path of Aircraft (Meters)")
    plt.legend()
    plt.grid(True)
    plt.xticks(range(-300, 301, 30))  
    plt.yticks(range(-300, 301, 30))  
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.plot(altitudes, marker='o', color='orange', label="Altitude")
    plt.xlabel("Step")
    plt.ylabel("Altitude (m)")
    plt.title("Altitude Variation")
    plt.grid(True)
    plt.show()


plot_path_with_target_and_quadrants("best_individual_log.txt")

