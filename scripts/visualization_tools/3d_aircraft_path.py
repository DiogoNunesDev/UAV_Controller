import matplotlib.pyplot as plt
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D

def haversine(lat1, lon1, lat2, lon2):
  """
  Calculate the great-circle distance between two points on Earth (in meters).
  """
  R = 6371000  # Earth radius in meters
  phi1, phi2 = np.radians([lat1, lat2])
  dphi = np.radians(lat2 - lat1)
  dlambda = np.radians(lon2 - lon1)
  a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
  c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
  return R * c

def lat_lon_to_meters(lat1, lon1, lat2, lon2):
  """
  Converts latitude and longitude differences to meters.
  """
  earth_radius = 6378137.0  # Earth radius in meters
  delta_lat = np.radians(lat2 - lat1)
  delta_lon = np.radians(lon2 - lon1)
  x = delta_lon * earth_radius * np.cos(np.radians(lat1))
  y = delta_lat * earth_radius
  return x, y

def parse_flight_data(file_path):
  """
  Parses the log file to extract aircraft position and orientation.
  """
  data = []
  with open(file_path, 'r') as file:
    for line in file:
      line = line.strip()
      if line and line[0].isdigit():  
        parts = line.split('\t')
        if len(parts) >= 7:  
          data.append({
            'step': int(parts[0]),
            'latitude': float(parts[1]),
            'longitude': float(parts[2]),
            'altitude': float(parts[3]),
            'heading': float(parts[4]),   # Yaw (degrees)
            'roll': float(parts[5]),      # Roll (degrees)
            'pitch': float(parts[6])      # Pitch (degrees)
          })
  return data

def animate_3d_trajectory(data):
  """
  Animates the 3D trajectory of the aircraft with real-time movement.
  """
  if not data:
      print("No flight data available.")
      return

  lat0, lon0 = data[0]['latitude'], data[0]['longitude']
  x_vals, y_vals, z_vals = [], [], []

  fig = plt.figure(figsize=(10, 7))
  ax = fig.add_subplot(111, projection='3d')

  x_min, y_min = lat_lon_to_meters(lat0, lon0, min(d['latitude'] for d in data), min(d['longitude'] for d in data))
  x_max, y_max = lat_lon_to_meters(lat0, lon0, max(d['latitude'] for d in data), max(d['longitude'] for d in data))
  z_min = min(d['altitude'] for d in data)
  z_max = max(d['altitude'] for d in data)

  ax.set_xlim(x_min, x_max)
  ax.set_ylim(y_min, y_max)
  ax.set_zlim(z_min, z_max)

  ax.set_xlabel("X (meters)")
  ax.set_ylabel("Y (meters)")
  ax.set_zlabel("Altitude (meters)")
  ax.set_title("Aircraft 3D Flight Path Animation")

  for i in range(len(data)):
    x, y = lat_lon_to_meters(lat0, lon0, data[i]['latitude'], data[i]['longitude'])
    z = data[i]['altitude']

    x_vals.append(x)
    y_vals.append(y)
    z_vals.append(z)

    ax.clear()
    ax.plot(x_vals, y_vals, z_vals, label="Aircraft Path", color='b')

    #Plotting aircraft position with orientation arrow
    if i > 0:
      dx = x_vals[i] - x_vals[i-1]
      dy = y_vals[i] - y_vals[i-1]
      dz = z_vals[i] - z_vals[i-1]

      norm = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-6
      dx /= norm
      dy /= norm
      dz /= norm

      ax.quiver(x_vals[i], y_vals[i], z_vals[i], dx, dy, dz, color='r', length=20, normalize=True)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_zlabel("Altitude (meters)")
    ax.set_title("Aircraft 3D Flight Path Animation")

    plt.pause(0.05)  

  plt.show()


file_path = "../../txt_files/ppo_log.txt" 
flight_data = parse_flight_data(file_path)
animate_3d_trajectory(flight_data)
