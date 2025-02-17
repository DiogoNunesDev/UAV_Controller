import random
import numpy as np
from gym_jsbsim.environment import JsbSimEnv
from gym_jsbsim.tasks import NavigationTask  
from gym_jsbsim.aircraft import cessna172P
import gym_jsbsim.properties as prp
import gc
import math
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback


STEP_FREQUENCY_HZ = 5  # Frequency at which actions are sent
EPISODE_TIME_S = 10  # Total episode duration in seconds
EARTH_RADIUS = 6371000  # Earth radius in meters
CIRCLE_RADIUS = 500     # Circle radius in meters
NUM_POINTS = 15         # Number of points on the circumference
TOLERANCE_DISTANCE = 10 # Tolerance distance in meters
ALTITUDE_THRESHOLD = 100  # Altitude threshold to detect crash or failure
START_LAT = 37.619
START_LON = -122.3750
TOTAL_TIMESTEPS = 10000000

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
    
def generate_equally_spaced_target_points(n_points, radius):
  """
  Generates `n` equally spaced points on a circle.
  """
  points = []
  random_offset = np.random.uniform(0, 2 * np.pi) 
  angle_increment = 2 * np.pi / n_points
  
  for i in range(n_points):
      angle = random_offset + i * angle_increment
      x = radius * np.cos(angle)
      y = radius * np.sin(angle)
      points.append((x, y))  

  return points

def create_target_points(start_lat, start_lon, radius, n_points):
  """
  Creates `n` equally spaced target points around a circle centered at the
  provided (start_lat, start_lon), using a given radius in meters.
  """
  circle_points = generate_equally_spaced_target_points(n_points, radius)
  
  target_points = []
  for point in circle_points:
      x, y = point
      angle = np.degrees(np.arctan2(y, x))
      target_lat, target_lon = calculate_circle_point(start_lat, start_lon, radius, angle)
      target_points.append((target_lat, target_lon))
  
  return target_points

if __name__ == "__main__":
  
  target_points = create_target_points(START_LAT, START_LON, CIRCLE_RADIUS, 50)
  target_point = random.choice(target_points)
  print(f"Training on Target Point: {target_point}")

  env = create_env(target_point)

  vec_env = DummyVecEnv([lambda: env])
  
  model_path = "../models/ppo_navigation_single_target.zip"
  
  try:
    model = PPO.load(model_path, env=vec_env)
    print("Loaded existing model.")
  
  except Exception as e:
      print("No saved model found, starting a new one.")
      model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=4096,
        batch_size=128,
        gae_lambda=0.95,
        gamma=0.99,
        verbose=1,
      )
  

  checkpoint_callback = CheckpointCallback(save_freq=0.25*TOTAL_TIMESTEPS, save_path='../models/', name_prefix='ppo_navigation_intermedio')

  print("Starting PPO training...")
  model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback)
  print("Training completed!")

  model.save("../models/ppo_navigation_single_target")
  print("Model saved as 'ppo_navigation_single_target'.")

  vec_env.close()
  gc.collect()

  print("Training completed for all target points!")
  