import random
import numpy as np
import math
import gc
import time
from gym_jsbsim.environment import JsbSimEnv
from gym_jsbsim.tasks import NavigationTask  
from gym_jsbsim.aircraft import cessna172P
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

# Constants
STEP_FREQUENCY_HZ = 5  # Frequency at which actions are sent
EPISODE_TIME_S = 10  # Total episode duration in seconds
EARTH_RADIUS = 6371000  # Earth radius in meters
CIRCLE_RADIUS = 500     # Circle radius in meters
NUM_POINTS = 15         # Number of points on the circumference
TOLERANCE_DISTANCE = 10 # Tolerance distance in meters
ALTITUDE_THRESHOLD = 100  # Altitude threshold to detect crash or failure
START_LAT = 37.619
START_LON = -122.3750
TOTAL_TIMESTEPS = 25000000
RESTART_INTERVAL = 2500000  # Restart JSBSim every 2.5M timesteps
MODEL_PATH = ""#"../models/ppo_navigation_single_target.zip"
SAVE_PATH = "../models/ppo_navigation"

def create_env(target_point):
    """Sets up the GymJSBSim environment for the navigation task."""
    return JsbSimEnv(
        task_type=NavigationTask,
        aircraft=cessna172P,
        agent_interaction_freq=STEP_FREQUENCY_HZ,
        shaping=None,
        target_point=target_point
    )

def calculate_circle_point(lat, lon, radius, angle):
    """Calculates a point on a circle given the center and radius."""
    lat_rad, lon_rad, angle_rad = map(math.radians, (lat, lon, angle))
    new_lat_rad = math.asin(math.sin(lat_rad) * math.cos(radius / EARTH_RADIUS) +
                            math.cos(lat_rad) * math.sin(radius / EARTH_RADIUS) * math.cos(angle_rad))
    new_lon_rad = lon_rad + math.atan2(math.sin(angle_rad) * math.sin(radius / EARTH_RADIUS) * math.cos(lat_rad),
                                        math.cos(radius / EARTH_RADIUS) - math.sin(lat_rad) * math.sin(new_lat_rad))
    return math.degrees(new_lat_rad), math.degrees(new_lon_rad)

def create_target_points(start_lat, start_lon, radius, n_points):
    """Creates n equally spaced target points around a circle."""
    angles = np.linspace(0, 360, n_points, endpoint=False)
    return [calculate_circle_point(start_lat, start_lon, radius, angle) for angle in angles]

if __name__ == "__main__":
  target_points = create_target_points(START_LAT, START_LON, CIRCLE_RADIUS, NUM_POINTS)
  target_point = random.choice(target_points)
  print(f"Training on Target Point: {target_point}")
  
  env = create_env(target_point)
  vec_env = DummyVecEnv([lambda: env])
  
  try:
      model = PPO.load(MODEL_PATH, env=vec_env, device="cpu", verbose=1)
      print("Loaded existing model.")
  except Exception:
      print("No saved model found, starting a new one.")
      model = PPO(
          "MlpPolicy",
          vec_env,
          learning_rate=3e-4,
          n_steps=4096,
          batch_size=128,
          gae_lambda=0.95,
          gamma=0.99,
          device="cpu",
          verbose=1,
      )
  
  # Training loop with JSBSim restart every RESTART_INTERVAL steps
  for i in range(TOTAL_TIMESTEPS // RESTART_INTERVAL):
      print(f"Starting training cycle {i+1}/{TOTAL_TIMESTEPS // RESTART_INTERVAL}")
      
      model.learn(total_timesteps=RESTART_INTERVAL)
      model.save(f"{SAVE_PATH}_latest")
      print(f"Model checkpoint saved at {SAVE_PATH}_latest")

      # Close and restart JSBSim
      vec_env.close()
      del vec_env
      del env
      gc.collect()
      print("Restarting JSBSim...")
      time.sleep(10)

      # Create new environment
      target_point = random.choice(target_points)  # Change target for variety
      env = create_env(target_point)
      vec_env = DummyVecEnv([lambda: env])
      model.set_env(vec_env)
      
  print("Training complete!")
  model.save(f"{SAVE_PATH}_final")
  print(f"Final model saved at {SAVE_PATH}_final")
  vec_env.close()
  gc.collect()
