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

def plot_path_with_target_and_quadrants(filename="best_individual_log.txt"):
    latitudes = []
    longitudes = []
    altitudes = []
    target_latitudes = []
    target_longitudes = []
    target_altitudes = []
    
    with open(filename, "r") as file:
        lines = file.readlines()

        for line in lines:
            if line.startswith("Target Latitude"): 
                target_line = line.strip().split(", ")
                if len(target_line) == 3:
                    try:
                        target_lat = float(target_line[0].split(": ")[1])
                        target_lon = float(target_line[1].split(": ")[1])
                        target_alt = float(target_line[2].split(": ")[1].replace("m", ""))
                        target_latitudes.append(target_lat)
                        target_longitudes.append(target_lon)
                        target_altitudes.append(target_alt)
                    except (IndexError, ValueError) as e:
                        print(f"Error processing target line: {line}. Error: {e}")
            elif line.startswith("Step"): 
                continue 
            elif line.strip(): 
                parts = line.strip().split("\t")
                if len(parts) == 4:
                    _, lat, lon, alt = parts
                    latitudes.append(float(lat))
                    longitudes.append(float(lon))
                    altitudes.append(float(alt))

    if not target_latitudes or not target_longitudes:
        print("No target data found in the file.")
        return

    segment_size = 50
    num_segments = (len(latitudes) + segment_size - 1) // segment_size 

    # Manually define colors for each segment
    colors = ['#3ba9f5', '#8405f7', '#0ffa0f']  # Blue, Purple, Dark Green

    plt.figure(figsize=(16, 12))

    for i in range(min(3, num_segments)):  
        start_idx = i * segment_size
        end_idx = min((i + 1) * segment_size, len(latitudes))

        # Plot path with specific color for each segment
        plt.subplot(3, 2, 2*i+1)
        plt.plot(longitudes[start_idx:end_idx], latitudes[start_idx:end_idx], marker='o', label=f"Path Target Point {i+1}", color=colors[i])
        plt.scatter(longitudes[start_idx], latitudes[start_idx], color="green", s=100, label="Start Point")
        plt.scatter(target_longitudes[i], target_latitudes[i], color="purple", s=100, label=f"Target Point {i+1}")
        plt.scatter(longitudes[end_idx-1], latitudes[end_idx-1], color="red", s=100, label="End Point")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(f"Path to Target Point {i+1}")
        plt.legend()
        plt.grid(True)

        # Plot altitude with specific color for each segment
        plt.subplot(3, 2, 2*i+2)
        steps = list(range(start_idx + 1, end_idx + 1))  
        plt.plot(steps, altitudes[start_idx:end_idx], marker='o', label=f"Altitude Variation {i+1}", color=colors[i])
        plt.xlabel("Step")
        plt.ylabel("Altitude (m)")
        plt.title(f"Altitude Variation {i+1}")
        plt.grid(True)

    plt.tight_layout()
    plt.show()

plot_path_with_target_and_quadrants("best_individual_log.txt")
