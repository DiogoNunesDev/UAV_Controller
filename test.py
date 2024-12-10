import gym
import gym_jsbsim
import numpy as np
from gym_jsbsim.simulation import Simulation
from gym_jsbsim.tasks import NavigationTask  
from gym_jsbsim.environment import JsbSimEnv
from gym_jsbsim.aircraft import Aircraft, cessna172P
from gym_jsbsim.visualiser import FlightGearVisualiser
import gym_jsbsim.properties as prp
import time
import math
import random
from tensorflow.keras.models import load_model



STEP_FREQUENCY_HZ = 5  # Frequency at which actions are sent
EPISODE_TIME_S = 60  # Total episode duration in seconds
RESET_AFTER_EPISODES = 3
EARTH_RADIUS = 6371000  # Earth radius in meters
START_LAT = 37.619
START_LON = -122.3750
MIN_MAX_RANGES = {
  'Pitch': (0, 2 * np.pi),          # Pitch range (degrees)
  'Roll': (0, 2 * np.pi),         # Roll range (degrees)
  'Yaw': (0, 360),             # Yaw range (degrees)
  'Throttle': (0, 1),          # Throttle range
  'Altitude': (0, 250),      # Example Altitude Distance range (meters)
  'Distance': (0, 1000),       # Example Distance range (meters)
  'Yaw Angle': (0, 360),  # Yaw Angle (radians)
  'Pitch Angle': (-90, 90)# Pitch Angle (radians)
}

model_path = "./best_model.h5"

def create_env():
    
    env = JsbSimEnv(
        task_type=NavigationTask,
        aircraft=cessna172P,
        agent_interaction_freq=STEP_FREQUENCY_HZ,
        shaping=None,  
        target_point=reset_target_point(START_LAT, START_LON)
    )

    """
    properties_to_display = [
        prp.altitude_agl_ft, prp.roll_rad, prp.pitch_rad,
        prp.heading_deg, prp.throttle, prp.aileron_cmd,
        prp.elevator_cmd, prp.rudder_cmd
    ]
    """
    return env

def calculate_circle_point(lat, lon, radius, angle):
    lat_rad, lon_rad, angle_rad = map(math.radians, (lat, lon, angle))

    new_lat_rad = math.asin(math.sin(lat_rad) * math.cos(radius / EARTH_RADIUS) +
                                math.cos(lat_rad) * math.sin(radius / EARTH_RADIUS) * math.cos(angle_rad))
    new_lon_rad = lon_rad + math.atan2(math.sin(angle_rad) * math.sin(radius / EARTH_RADIUS) * math.cos(lat_rad),
                                           math.cos(radius / EARTH_RADIUS) - math.sin(lat_rad) * math.sin(new_lat_rad))
    return math.degrees(new_lat_rad), math.degrees(new_lon_rad)

def reset_target_point(start_lat, start_lon, radius = 250):
    
    # Generates points around the circle at the given radius and selects a random one
    circle_points = [calculate_circle_point(start_lat, start_lon, radius, i * (360 / 15)) for i in range(15)]
    target_point = random.choice(circle_points)
    return target_point

def normalize_input_vector(input_vector):
    normalized_vector_min_max = []
    feature_names = ['Pitch', 'Roll', 'Yaw', 'Throttle', 'Altitude', 'Distance', 'Yaw Angle', 'Pitch Angle']

    for i, feature in enumerate(input_vector):
        feature_name = feature_names[i]
        
        if feature_name in MIN_MAX_RANGES:
            feature_range = MIN_MAX_RANGES[feature_name]
            normalized_feature = (feature - feature_range[0]) / (feature_range[1] - feature_range[0])
            normalized_vector_min_max.append(normalized_feature)
        else:
            normalized_vector_min_max.append(feature)
    
    return normalized_vector_min_max

def run_random_controller():
    env = create_env()
    episode_count = 0    
    model = load_model(model_path, compile=False)
    while episode_count < RESET_AFTER_EPISODES:
        obs = env.reset()
        env.render()
        done = False
        step_count = 0
        
        time.sleep(20)
        print(f"Starting episode {episode_count + 1}")

        # Main loop for each episode
        while not done:
            normalized_input_vector = normalize_input_vector(obs)
            input_vector = np.array(normalized_input_vector).reshape(1, -1)
            action = model.predict(input_vector, verbose=0)[0]
            roll, pitch, yaw = action[:3]
            throttle = (action[3] + 1) / 2
            action = np.array([roll, pitch, yaw, throttle])
            obs, reward, done, info = env.step(action)
            step_count += 1
            
            print(f"Step: {step_count}, Reward: {reward}, Done: {done}")
            print(f"Observation: {obs}")
            print(f'Info: {info}')
            
            time.sleep(0.2) 

        episode_count += 1
        print(f"Episode {episode_count} finished.\n")

    print("All episodes completed.")
    env.close()

run_random_controller()