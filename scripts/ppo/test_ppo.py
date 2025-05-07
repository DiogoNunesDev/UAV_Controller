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
from gym import spaces
import time

STEP_FREQUENCY_HZ = 60  # Frequency at which actions are sent
EPISODE_TIME_S = 240  # Total episode duration in seconds
EARTH_RADIUS = 6371000  # Earth radius in meters
CIRCLE_RADIUS = 4500     # Circle radius in meters
START_LAT = 37.619
START_LON = -122.3750

#STATE SPACE
lows = np.array([
    -np.pi,   # Roll
    -np.pi/2, # Pitch
    -np.pi,   # Yaw (normalized to -π to π)
    #0.0,      # Throttle (0 to 1)
    -205,      # Altitude Deviation (meters)
    #0.0,      # Distance (meters)
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
    #1.0,      # Throttle
    305,     # Altitude Deviation(meters)
    #10000,     # Distance
    np.pi,      # Yaw angle
    np.pi/2,        # Pitch angle
    2200,
    250,
    2 * math.pi,
    2 * math.pi,
    2 * math.pi,
], dtype=np.float32)


def unnormalize_observation(normalized_obs: np.ndarray) -> np.ndarray:
  """
  Reverts the normalization of an observation from [-1, 1] back to the original scale.
  """
  # Reverse min-max scaling: x = 0.5 * ((x_norm + 1) * (max - min)) + min
  original_obs = 0.5 * (normalized_obs + 1) * (highs - lows) + lows

  return original_obs

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
    target_points = create_target_points(START_LAT, START_LON, n=3)
    target_point = random.choice(target_points)
    print(f"Testing on Target Point: {target_point}")

    env = create_env(target_point)
    model = PPO.load("../models/ppo_navigation/ppo_navigation_6399744_steps.zip")
    
    obs = env.reset()
    log = []
    log.append(f"Target Latitude: {target_point[0]}, Target Longitude: {target_point[1]}, Target Altitude: 300m")
    log.append("Step\tLatitude\tLongitude\tAltitude\tHeading\tRoll\tPitch")
    
    done = False
    step_count = 0
    observations = []
    throttle_values = []
    altitude_values = []
    #env.render(mode='flightgear')
    #time.sleep(20)
    while not done:#EPISODE_TIME_S * STEP_FREQUENCY_HZ:
        action, _states = model.predict(obs, deterministic=True)
        #action[3] = 1.0
        #action[3] += 0.5
        obs, reward, done, info = env.step(action)
        #env.render()
        #print(f"Reward: {reward}")
        #print(action[3])
        #print(_states)
        unnormalize_obs = unnormalize_observation(obs)
        #print([round(val, 2) for val in unnormalize_obs])
        #print("")
        print(f"Engine: {env.sim[prp.engine_running]}")
        observations.append(list(unnormalize_obs))
        step_count += 1
        current_lat = env.sim[prp.lat_geod_deg]
        current_lon = env.sim[prp.lng_geoc_deg]
        current_alt = env.sim[prp.altitude_agl_ft] * 0.3048
        altitude_values.append([current_alt, unnormalize_obs[3]])
        throttle = env.sim[prp.throttle_cmd]
        throttle_values.append(throttle)
        #print(f"Predicted Throttle: {action[3]}")
        print(f"Throttle: {throttle}")
        #observations.append(throttle)
        heading = unnormalize_obs[2]
        roll = unnormalize_obs[0]
        pitch = unnormalize_obs[1]
        if step_count % 10 == 0:
            log.append(f"{step_count}\t{current_lat}\t{current_lon}\t{current_alt}\t{heading}\t{roll}\t{pitch}")
    
    save_logs(log)
    gc.collect()
    
    obs_labels = [
        "Roll (deg)", "Pitch (deg)", "Yaw (deg)", #"Throttle",
        "Altitude Deviation (m)", #"Distance to Target (m)",
        "Yaw Angle to Target (deg)", "Pitch Angle to Target (deg)", 
        "Velocity X-Axis", "Altitude_change", "Pitch Rate [deg/s]", "Yaw Rate [deg/s]", "Roll Rate [deg/s]",
    ]
    
    df_obs = pd.DataFrame(observations, columns=obs_labels)
    #df_obs["Throttle"] = df_obs["Throttle"] * 100
    df_obs["Roll (deg)"] = np.degrees(df_obs["Roll (deg)"])
    df_obs["Pitch (deg)"] = np.degrees(df_obs["Pitch (deg)"])
    df_obs["Yaw (deg)"] = np.degrees(df_obs["Yaw (deg)"])
    df_obs["Yaw Angle to Target (deg)"] = np.degrees(df_obs["Yaw Angle to Target (deg)"])
    df_obs["Pitch Angle to Target (deg)"] = np.degrees(df_obs["Pitch Angle to Target (deg)"])
    df_obs["Pitch Rate [deg/s]"] = np.degrees(df_obs["Pitch Rate [deg/s]"])
    df_obs["Yaw Rate [deg/s]"] = np.degrees(df_obs["Yaw Rate [deg/s]"])
    df_obs["Roll Rate [deg/s]"] = np.degrees(df_obs["Roll Rate [deg/s]"])

    
    save_csv(df_obs)
    
    plt.figure(figsize=(12, 8))
    for column in df_obs.columns:
        plt.plot(df_obs.index.to_numpy(), df_obs[column].to_numpy(), label=column)
    
    plt.xlabel("Timestep")
    plt.ylabel("Value")
    plt.title("PPO Observations Over Time (Angles in Degrees)")
    plt.legend()
    plt.grid(True)
    plt.savefig("sensores.png")
    plt.show()
    
    altitude_array = np.array(altitude_values)

    plt.figure(figsize=(12, 8))
    plt.plot(altitude_array[:, 0], label="Altitude")
    plt.plot(altitude_array[:, 1], label="Altitude Deviation")
    plt.xlabel("Timestep")
    plt.ylabel("Value")
    plt.title("Altitude & ALtitude Deviation")
    plt.legend()
    plt.grid(True)
    plt.savefig("altitude.png")
    plt.show()
    
    plt.figure(figsize=(12, 8))
    plt.plot(np.array(throttle_values), label="Throttle")
    
    plt.xlabel("Timestep")
    plt.ylabel("Value")
    plt.title("Throttle")
    plt.legend()
    plt.grid(True)
    plt.savefig("throttle.png")
    plt.show()
