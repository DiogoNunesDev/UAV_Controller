import random
import numpy as np
import math
import gc
import time
import os
from gym_jsbsim.environment import JsbSimEnv
from gym_jsbsim.tasks import NavigationTask  
from gym_jsbsim.aircraft import cessna172P
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

# Constants
STEP_FREQUENCY_HZ = 5  
EPISODE_TIME_S = 10  
EARTH_RADIUS = 6371000  
CIRCLE_RADIUS = 500     
NUM_POINTS = 15         
TOLERANCE_DISTANCE = 10 
ALTITUDE_THRESHOLD = 100  
START_LAT = 37.619
START_LON = -122.3750
TOTAL_TIMESTEPS = 50000000
RESTART_INTERVAL = 5000000  
MODEL_PATH = ""#"../models/ppo_50M"  
SAVE_PATH = "../models/ppo_navigation"

NUM_CPU = 5

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

def make_env(target_point):
    """Function to create an individual JSBSim environment for multiprocessing."""
    def _init():
        return JsbSimEnv(
            task_type=NavigationTask,
            aircraft=cessna172P,
            agent_interaction_freq=STEP_FREQUENCY_HZ,
            shaping=None,
            target_point=target_point
        )
    return _init

if __name__ == "__main__":
    target_points = create_target_points(START_LAT, START_LON, CIRCLE_RADIUS, NUM_POINTS)

    def create_vec_env():
        """Creates a vectorized environment using multiple CPU processes."""
        return SubprocVecEnv([make_env(random.choice(target_points)) for _ in range(NUM_CPU)])

    print("Initializing environments...")
    vec_env = create_vec_env()

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=RESTART_INTERVAL,
        save_path=os.path.dirname(SAVE_PATH),
        name_prefix="ppo_navigation",
        save_replay_buffer=True,
        save_vecnormalize=True
    )

    try:
        model = PPO.load(MODEL_PATH, env=vec_env, device="cpu", verbose=1)
        print("Loaded existing model.")
        
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback, reset_num_timesteps=False)
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
        
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback)

    print("Training complete!")
    model.save(f"{SAVE_PATH}_final")
    print(f"Final model saved at {SAVE_PATH}_final")
    vec_env.close()
    gc.collect()
