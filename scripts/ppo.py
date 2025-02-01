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
TOTAL_TIMESTEPS = 50000

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

def evaluate_individuals(individuals, input_dim, output_dim, target_points):
  """Evaluates a batch of individuals in the provided environment instance."""
  run_index = 1
  for target_point in target_points:
    env = create_env(target_point)
    results = []
    for individual in individuals:
      model = NeuralNetwork(input_dim, output_dim).genome_to_model(individual.genome)
      obs = env.reset()
      input_obs = obs[0:8]
      done, step_count, total_time, crashed = False, 0, 0, False
      cumulative_altitude_dist = 0
      individual.log.append(f"Target Latitude: {env.task.target_point[0]:.6f}, Target Longitude: {env.task.target_point[1]:.6f}, Target Altitude: 300m")
      individual.log.append("Step\tLatitude\tLongitude\tAltitude\tHeading")
      
      while not done and step_count < EPISODE_TIME_S * STEP_FREQUENCY_HZ:
        normalized_input_vector = normalize_input_vector(input_obs)
        input_vector = np.array(normalized_input_vector).reshape(1, -1)
        individual.pry.append(f"Pitch (deg): {math.degrees(obs[0])}, Roll (deg): {math.degrees(obs[1])}, Yaw (deg): {math.degrees(obs[2])}")
        action = model.predict(input_vector, verbose=0)[0]
        roll, pitch, yaw = action[:3]
        throttle = (action[3] + 1) / 2
        action = np.array([roll, pitch, yaw, throttle])

        obs, reward, done, info = env.step(action)
        step_count += 1
        current_lat = obs[9]
        current_lon = obs[10]
        current_alt = obs[4]
        cumulative_altitude_dist += (abs(300 - current_alt))
        crashed = current_alt <= ALTITUDE_THRESHOLD
        individual.ardupilot_log[run_index].append(obs)
        individual.log.append(f"{step_count}\t{current_lat:.6f}\t{current_lon:.6f}\t{current_alt}\t{math.degrees(obs[2])}")
        input_obs = obs[0:8]
 
      distance_to_target = info.get('distance_to_target', float('inf'))
      results.append((distance_to_target, crashed, step_count, cumulative_altitude_dist, individual))
    
    run_index += 1

  print("All individuals were evaluated!")
  env.close()
  return results    


if __name__ == "__main__":
  
  target_points = create_target_points(START_LAT, START_LON, n=3)
  target_point = target_points[1]
  print(f"Training on Target Point: {target_point}")

  env = create_env(target_point)

  vec_env = DummyVecEnv([lambda: env])

  model = PPO(
      "MlpPolicy",
      vec_env,
      learning_rate=3e-4,
      n_steps=2048,
      batch_size=64,
      gae_lambda=0.95,
      gamma=0.99,
      verbose=1,
  )

  print("Starting PPO training...")
  model.learn(total_timesteps=TOTAL_TIMESTEPS)
  print("Training completed!")

  model.save("ppo_navigation_single_target")
  print("Model saved as 'ppo_navigation_single_target'.")

  vec_env.close()
  gc.collect()

  print("Training completed for all target points!")
  