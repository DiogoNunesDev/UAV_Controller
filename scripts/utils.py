import numpy as np
import math
from gym import spaces
from geopy.distance import distance

def get_state_space():
  lows = np.array([
      -np.pi,   # Roll
      -np.pi/2, # Pitch
      -np.pi,   # Yaw (normalized to -π to π)
      0.0,      # Throttle (0 to 1)
      0.0,      # Altitude (meters)
      0.0,      # Distance (meters)
      -np.pi,     # Yaw angle (degrees)
      -np.pi/2,       # Pitch angle (degrees)
      -2200,
      -250,
      -2 * math.pi, 
      -2 * math.pi,
      -2 * math.pi,
  ], dtype=np.float32)

  highs = np.array([
      np.pi,    # Roll
      np.pi/2,  # Pitch
      np.pi,    # Yaw
      1.0,      # Throttle
      1000,     # Altitude (meters)
      3000,     # Distance
      np.pi,      # Yaw angle
      np.pi/2,        # Pitch angle
      2200,
      250,
      2 * math.pi,
      2 * math.pi,
      2 * math.pi,
  ], dtype=np.float32)
  return spaces.Box(low=lows, high=highs, dtype=np.float32)

def unnormalize_observation(normalized_obs: np.ndarray) -> np.ndarray:
  """
  Reverts the normalization of an observation from [-1, 1] back to the original scale.
  """
  lows = get_state_space().low
  highs = get_state_space().high

  # Reverse min-max scaling: x = 0.5 * ((x_norm + 1) * (max - min)) + min
  original_obs = 0.5 * ((normalized_obs + 1) * (highs - lows)) + lows

  return original_obs

def calculate_circle_point_geopy(lat, lon, radius, angle):
  """
  Uses geopy to compute points on a geodesic circle given a center (lat, lon),
  a radius (in meters), and an angle in degrees.
  Returns latitude and longitude as a tuple.
  """
  destination = distance(meters=radius).destination((lat, lon), angle)
  return destination.latitude, destination.longitude

def create_target_points_geopy(start_lat, start_lon, radius, n=10):
  """
  Creates `n` equally spaced target points around a circle centered at (start_lat, start_lon).
  """
  angles = np.linspace(0, 360, n, endpoint=False)  # Generate equally spaced angles
  return [calculate_circle_point_geopy(start_lat, start_lon, radius, angle) for angle in angles]

