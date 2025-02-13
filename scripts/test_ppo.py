import numpy as np
from gym_jsbsim.environment import JsbSimEnv
from gym_jsbsim.tasks import NavigationTask  
from gym_jsbsim.aircraft import cessna172P
import gym_jsbsim.properties as prp
import gc
import math
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


STEP_FREQUENCY_HZ = 5  # Frequency at which actions are sent
EPISODE_TIME_S = 10  # Total episode duration in seconds
EARTH_RADIUS = 6371000  # Earth radius in meters
CIRCLE_RADIUS = 250     # Circle radius in meters
NUM_POINTS = 15         # Number of points on the circumference
TOLERANCE_DISTANCE = 10 # Tolerance distance in meters
ALTITUDE_THRESHOLD = 100  # Altitude threshold to detect crash or failure
START_LAT = 37.619
START_LON = -122.3750
TOTAL_TIMESTEPS = 100000

MIN_MAX_RANGES = {
  'Pitch': (0, 2 * np.pi),    #Pitch range (degrees)
  'Roll': (0, 2 * np.pi),     #Roll range (degrees)
  'Yaw': (0, 360),            #Yaw range (degrees)
  'Throttle': (0, 1),         #Throttle range
  'Altitude': (0, 250),       #Altitude Distance range (meters)
  'Distance': (0, 1000),      #Distance range (meters)
  'Yaw Angle': (-180, 180),   #Yaw Angle (radians)
  'Pitch Angle': (-90, 90)    #Pitch Angle (radians)
}


def create_env(target_point):
  """Sets up the GymJSBSim environment for the navigation task."""
  env = JsbSimEnv(
    task_type=NavigationTask,
    aircraft=cessna172P,
    agent_interaction_freq=STEP_FREQUENCY_HZ,
    shaping=None,
    target_point=target_point
  )
  return env

def calculate_circle_point(lat, lon, radius, angle):
  """
  This function calculates a point on the surface of the Earth
  """
  lat_rad, lon_rad, angle_rad = map(math.radians, (lat, lon, angle))

  new_lat_rad = math.asin(math.sin(lat_rad) * math.cos(radius / EARTH_RADIUS) +
                          math.cos(lat_rad) * math.sin(radius / EARTH_RADIUS) * math.cos(angle_rad))
  new_lon_rad = lon_rad + math.atan2(math.sin(angle_rad) * math.sin(radius / EARTH_RADIUS) * math.cos(lat_rad),
                                      math.cos(radius / EARTH_RADIUS) - math.sin(lat_rad) * math.sin(new_lat_rad))
  return math.degrees(new_lat_rad), math.degrees(new_lon_rad)
    
def generate_equally_spaced_target_points(n=3, radius=CIRCLE_RADIUS):
  """
  Generates `n` equally spaced points on a circle.
  """
  points = []
  random_offset = np.random.uniform(0, 2 * np.pi) 
  angle_increment = 2 * np.pi / n 
  
  for i in range(n):
      angle = random_offset + i * angle_increment
      x = radius * np.cos(angle)
      y = radius * np.sin(angle)
      points.append((x, y))  

  return points

def create_target_points(start_lat, start_lon, radius=CIRCLE_RADIUS, n=3):
  """
  Creates `n` equally spaced target points around a circle centered at the
  provided (start_lat, start_lon), using a given radius in meters.
  """
  circle_points = generate_equally_spaced_target_points(n, radius)
  
  target_points = []
  for point in circle_points:
      x, y = point
      angle = np.degrees(np.arctan2(y, x))
      target_lat, target_lon = calculate_circle_point(start_lat, start_lon, radius, angle)
      target_points.append((target_lat, target_lon))
  
  return target_points


def save_logs(log):
  with open("../txt_files/ppo_log.txt", "w") as log_file:
    
    log_file.write(f"Best Fitness: {0}\n")
    log_file.write(f"Target Latitude: {log[0].split(',')[0].split(':')[1].strip()}, ")
    log_file.write(f"Target Longitude: {log[0].split(',')[1].split(':')[1].strip()}, ")
    log_file.write(f"Target Altitude: 300m\n")  
            
    for step_log in log[1:]:  
      log_file.write(step_log + "\n")  

if __name__ == "__main__":
  
    target_points = create_target_points(START_LAT, START_LON, n=3)
    target_point = target_points[1]
    print(f"Training on Target Point: {target_point}")

    env = create_env(target_point)

    model = PPO.load("./ppo_navigation_single_target")

    obs = env.reset()  
    log = []
    log.append(f"Target Latitude: {env.task.target_point[0]:.6f}, Target Longitude: {env.task.target_point[1]:.6f}, Target Altitude: 300m")
    log.append("Step\tLatitude\tLongitude\tAltitude\tHeading")
    done = False
    step_count = 0

    while not done and step_count < EPISODE_TIME_S * STEP_FREQUENCY_HZ:
      action, _states = model.predict(obs, deterministic=True)
      obs, reward, done, info = env.step(action)
      step_count += 1

      current_lat = env.sim[prp.lat_geod_deg]
      current_lon = env.sim[prp.lng_geoc_deg]
      current_alt = env.sim[prp.altitude_agl_ft] * 0.3048
      heading = env.sim[prp.heading_deg]
      log.append(f"{step_count}\t{current_lat:.6f}\t{current_lon:.6f}\t{current_alt}\t{heading}")

    save_logs(log)
    gc.collect()

    print("Training completed for all target points!")
  