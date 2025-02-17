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
import pandas as pd
import matplotlib.pyplot as plt

STEP_FREQUENCY_HZ = 5  # Frequency at which actions are sent
EPISODE_TIME_S = 10  # Total episode duration in seconds
EARTH_RADIUS = 6371000  # Earth radius in meters
CIRCLE_RADIUS = 250     # Circle radius in meters
START_LAT = 37.619
START_LON = -122.3750


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
    
def generate_equally_spaced_target_points(n=5, radius=CIRCLE_RADIUS):
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

def create_target_points(start_lat, start_lon, radius=CIRCLE_RADIUS, n=5):
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
    with open("../../txt_files/ppo_log.txt", "w") as log_file:
        for step_log in log:
            log_file.write(step_log + "\n")


def save_csv(df, filename="ppo_observations.csv"):
    """Saves the observation DataFrame to a CSV file."""
    df.to_csv(filename, index=False)
    print(f"Observations saved to {filename}")


if __name__ == "__main__":
    target_points = create_target_points(START_LAT, START_LON, n=5)
    target_point = random.choice(target_points)
    print(f"Testing on Target Point: {target_point}")

    env = create_env(target_point)
    model = PPO.load("../models/ppo_navigation_intermedio_5000000_steps.zip")
    
    obs = env.reset()
    log = []
    log.append(f"Target Latitude: {target_point[0]}, Target Longitude: {target_point[1]}, Target Altitude: 300m")
    log.append("Step\tLatitude\tLongitude\tAltitude\tHeading")
    
    done = False
    step_count = 0
    observations = []
    
    while not done and step_count < 250:#EPISODE_TIME_S * STEP_FREQUENCY_HZ:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        print(obs)
        observations.append(list(obs))
        step_count += 1
        current_lat = env.sim[prp.lat_geod_deg]
        current_lon = env.sim[prp.lng_geoc_deg]
        current_alt = env.sim[prp.altitude_agl_ft] * 0.3048
        heading = env.sim[prp.heading_deg]
        log.append(f"{step_count}\t{current_lat}\t{current_lon}\t{current_alt}\t{heading}")
    
    save_logs(log)
    gc.collect()
    
    obs_labels = [
        "Roll (deg)", "Pitch (deg)", "Yaw (deg)", "Throttle",
        "Altitude (m)", "Distance to Target (m)",
        "Yaw Angle to Target (deg)", "Pitch Angle to Target (deg)"
    ]
    df_obs = pd.DataFrame(observations, columns=obs_labels)
    df_obs["Throttle"] = df_obs["Throttle"] * 100
    df_obs["Roll (deg)"] = np.degrees(df_obs["Roll (deg)"])
    df_obs["Pitch (deg)"] = np.degrees(df_obs["Pitch (deg)"])
    df_obs["Yaw (deg)"] = np.degrees(df_obs["Yaw (deg)"])
    df_obs["Yaw Angle to Target (deg)"] = np.degrees(df_obs["Yaw Angle to Target (deg)"])
    df_obs["Pitch Angle to Target (deg)"] = np.degrees(df_obs["Pitch Angle to Target (deg)"])
    
    save_csv(df_obs)
    
    plt.figure(figsize=(12, 8))
    for column in df_obs.columns:
        plt.plot(df_obs.index.to_numpy(), df_obs[column].to_numpy(), label=column)
    
    plt.xlabel("Timestep")
    plt.ylabel("Value")
    plt.title("PPO Observations Over Time (Angles in Degrees)")
    plt.legend()
    plt.grid(True)
    plt.show()
