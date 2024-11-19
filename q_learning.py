"""

  Q-LEARNING SETUP

  Agent: Aircraft
  State: Step variables of JSBSim
  Actions: V(Pitch, Roll, Yaw, Throttle)
  Rewards: 
          Negative: Crashing, Going out of stable altitude threshold
          Positive: Closer to target, Correct heading
          
  Q-values are calculated by: Q(s,a) = Q(s,a) + α * (r + γ * max(Q(s',a')) - Q(s,a))
                              
                              Q(s,a): Expected reward for taking action a in state s
                              α: learning rate
                              γ: discount factor
                              r: reward
                              s': new state
                              a': new action
                              max(Q(s',a')): Highest expected reward for all possible actions a' in the new state
                              
  Q-table: Rows: States
           Columns: Actions 
           
  State: V[ current_roll,
            current_pitch,
            current_yaw,
            throttle,
            current_alt,
            distance,
            yaw_angle,
            pitch_angle  ]

"""
from neural_network import NeuralNetwork
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
from gym_jsbsim.environment import JsbSimEnv
from gym_jsbsim.tasks import NavigationTask  
from gym_jsbsim.aircraft import cessna172P
import gym_jsbsim.properties as prp
import gc
from tensorflow.keras import backend as K

#CONSTANTS

STEP_FREQUENCY_HZ = 5  # Frequency at which actions are sent
EPISODE_TIME_S = 10  # Total episode duration in seconds
EARTH_RADIUS = 6371000  # Earth radius in meters
CIRCLE_RADIUS = 300     # Circle radius in meters
NUM_POINTS = 15         # Number of points on the circumference
TOLERANCE_DISTANCE = 10 # Tolerance distance in meters
ALTITUDE_THRESHOLD = 100  # Altitude threshold to detect crash or failure
NUM_THREADS = 3


pitch_values = np.round(np.arange(-1.0, 1.05, 0.05), 2)
roll_values = np.round(np.arange(-1.0, 1.05, 0.05), 2)
yaw_values = np.round(np.arange(-1.0, 1.05, 0.05), 2)
throttle_values = np.round(np.arange(0.0, 1.05, 0.05), 2)

combinations = np.array(np.meshgrid(pitch_values, roll_values, yaw_values, throttle_values)).T.reshape(-1, 4)
combinations_df = pd.DataFrame(combinations, columns=["Pitch", "Roll", "Yaw", "Throttle"])



def create_env():
  """Sets up the GymJSBSim environment for the navigation task."""
  env = JsbSimEnv(
    task_type=NavigationTask,
    aircraft=cessna172P,
    agent_interaction_freq=STEP_FREQUENCY_HZ,
    shaping=None,
  )
  return env

def main():
  env = create_env()
  obs = env.reset()
  done, step_count, total_time, crashed = False, 0, 0, False
  
  while not done and step_count < EPISODE_TIME_S * STEP_FREQUENCY_HZ:

      roll, pitch, yaw = action[:3]
      throttle = (action[3] + 1) / 2
      action = np.array([roll, pitch, yaw, throttle])

      obs, reward, done, info = env.step(action)
      step_count += 1
      total_time += 1 / STEP_FREQUENCY_HZ

      current_lat = env.sim[prp.lat_geod_deg]
      current_lon = env.sim[prp.lng_geoc_deg]
      current_alt = env.sim[prp.altitude_agl_ft]
      crashed = (current_alt * 0.3048) <= ALTITUDE_THRESHOLD

      #current_log.append(f"{step_count}\t{current_lat:.6f}\t{current_lon:.6f}\t{current_alt * 0.3048}")

      distance_to_target = info.get('distance_to_target', float('inf'))


  

def evaluate_individual(env, individual, input_dim, output_dim):
    """Evaluates a single individual in the provided environment instance."""
    model = NeuralNetwork(input_dim, output_dim).genome_to_model(individual.genome)
    obs = env.reset()
    done, step_count, total_time, crashed = False, 0, 0, False
        
    #current_log = []
    #current_log.append(f"Target Latitude: {env.task.target_point[0]:.6f}, Target Longitude: {env.task.target_point[1]:.6f}, Target Altitude: 300m")
    #current_log.append("Step\tLatitude\tLongitude\tAltitude")

    while not done and step_count < EPISODE_TIME_S * STEP_FREQUENCY_HZ:
      input_vector = np.array(obs).reshape(1, -1)
      action = model.predict(input_vector, verbose=0)[0]
      roll, pitch, yaw = action[:3]
      throttle = (action[3] + 1) / 2
      action = np.array([roll, pitch, yaw, throttle])

      obs, reward, done, info = env.step(action)
      step_count += 1
      total_time += 1 / STEP_FREQUENCY_HZ

      current_lat = env.sim[prp.lat_geod_deg]
      current_lon = env.sim[prp.lng_geoc_deg]
      current_alt = env.sim[prp.altitude_agl_ft]
      crashed = (current_alt * 0.3048) <= ALTITUDE_THRESHOLD

      #current_log.append(f"{step_count}\t{current_lat:.6f}\t{current_lon:.6f}\t{current_alt * 0.3048}")

    distance_to_target = info.get('distance_to_target', float('inf'))

    """
    with lock:
      if self.bestIndividual is None or current_fitness > self.bestIndividual.fitness:
        self.bestIndividual = individual
        with open("best_individual_log.txt", "w") as best_log:
          best_log.write(f"Best Fitness: {self.bestIndividual.fitness}\n")
          best_log.write("\n".join(current_log))
       
    """   
    return distance_to_target, total_time, crashed, step_count, current_alt, individual