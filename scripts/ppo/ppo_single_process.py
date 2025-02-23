import random
import numpy as np
import math
import gc
import time
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from gym_jsbsim.environment import JsbSimEnv
from gym_jsbsim.tasks import NavigationTask  
from gym_jsbsim.aircraft import cessna172P
import gym_jsbsim.properties as prp

# Constants
STEP_FREQUENCY_HZ = 5  
EPISODE_TIME_S = 10  
EARTH_RADIUS = 6371000  
CIRCLE_RADIUS = 250    
START_LAT = 37.619
START_LON = -122.3750
TOTAL_TIMESTEPS = 10000000  
RESTART_INTERVAL = 2500000  
SAVE_PATH = "../models/ppo_navigation"

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
    """Calculates a point on the surface of the Earth."""
    lat_rad, lon_rad, angle_rad = map(math.radians, (lat, lon, angle))
    new_lat_rad = math.asin(math.sin(lat_rad) * math.cos(radius / EARTH_RADIUS) +
                            math.cos(lat_rad) * math.sin(radius / EARTH_RADIUS) * math.cos(angle_rad))
    new_lon_rad = lon_rad + math.atan2(math.sin(angle_rad) * math.sin(radius / EARTH_RADIUS) * math.cos(lat_rad),
                                       math.cos(radius / EARTH_RADIUS) - math.sin(lat_rad) * math.sin(new_lat_rad))
    return math.degrees(new_lat_rad), math.degrees(new_lon_rad)

def create_target_points(start_lat, start_lon, radius=CIRCLE_RADIUS, n=5):
    """Creates n equally spaced target points around a circle centered at (start_lat, start_lon)."""
    angles = np.linspace(0, 360, n, endpoint=False)
    return [calculate_circle_point(start_lat, start_lon, radius, angle) for angle in angles]

def save_csv(df, filename="ppo_observations.csv"):
    """Saves the observation DataFrame to a CSV file."""
    df.to_csv(filename, index=False)
    print(f"Observations saved to {filename}")

class SaveModelCallback(CheckpointCallback):
    """Custom callback for saving models at regular intervals."""
    def __init__(self, save_freq, save_path, name_prefix="ppo_model"):
        super(SaveModelCallback, self).__init__(save_freq=save_freq, save_path=save_path, name_prefix=name_prefix)

    def _on_step(self) -> bool:
        return super()._on_step()

if __name__ == "__main__":
    target_points = create_target_points(START_LAT, START_LON, n=5)
    target_point = target_points[0]  # Keep the target constant
    print(f"Training on Fixed Target Point: {target_point}")

    # Create and wrap the environment
    env = DummyVecEnv([lambda: create_env(target_point)])

    # Load model or create a new one
    try:
        model = PPO.load(f"{SAVE_PATH}_latest", env=env, verbose=1)
        print("Loaded existing model.")
    except Exception:
        print("No saved model found, starting a new one.")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=4096,
            batch_size=128,
            gae_lambda=0.95,
            gamma=0.99,
            verbose=1,
            device="cpu",
        )

    # Callback to save the model at intervals
    callback = SaveModelCallback(save_freq=RESTART_INTERVAL, save_path=SAVE_PATH, name_prefix="ppo_model")

    # Start training without resetting the environment
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

    print("Training complete!")
    model.save(f"{SAVE_PATH}_final")
    print(f"Final model saved at {SAVE_PATH}_final")
