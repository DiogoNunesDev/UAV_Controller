from controller.scripts.genetic_algo.neural_network import NeuralNetwork
import random
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from gym_jsbsim.environment import JsbSimEnv
from gym_jsbsim.tasks import NavigationTask  
from gym_jsbsim.aircraft import cessna172P
import gym_jsbsim.properties as prp
import gc
from tensorflow.keras import backend as K
import tensorflow as tf
import math
from pymavlink.dialects.v20 import common as mavlink

#from sklearn.preprocessing import MinMaxScaler


#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


STEP_FREQUENCY_HZ = 5  # Frequency at which actions are sent
EPISODE_TIME_S = 10  # Total episode duration in seconds
EARTH_RADIUS = 6371000  # Earth radius in meters
CIRCLE_RADIUS = 250     # Circle radius in meters
NUM_POINTS = 15         # Number of points on the circumference
TOLERANCE_DISTANCE = 10 # Tolerance distance in meters
ALTITUDE_THRESHOLD = 100  # Altitude threshold to detect crash or failure
NUM_THREADS = 4
START_LAT = 37.619
START_LON = -122.3750

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


class Individual:
  def __init__(self, genome):
    self.genome = genome
    self.fitness = None
    self.log = []
    self.ardupilot_log = {1:[],2:[], 3:[]}
    self.pry = []
    
def evaluate_individual(env, individual, input_dim, output_dim, scaler):
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


class Genetic_Algorithm():

  def __init__(self, input_dim, output_dim, maxPopulation, generationMax, mutationProb, tournamentSize, elitismRate):
    self.nn = NeuralNetwork(input_dim, output_dim)
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.maxPopulation = maxPopulation
    self.generationMax = generationMax
    self.mutationProb = mutationProb
    self.tournamentSize = tournamentSize
    self.elitismRate = elitismRate
    self.population = []
    self.bestIndividual = None 
    
  def calculate_circle_point(self, lat, lon, radius, angle):
    """
    This function calculates a point on the surface of the Earth
    """
    lat_rad, lon_rad, angle_rad = map(math.radians, (lat, lon, angle))

    new_lat_rad = math.asin(math.sin(lat_rad) * math.cos(radius / EARTH_RADIUS) +
                            math.cos(lat_rad) * math.sin(radius / EARTH_RADIUS) * math.cos(angle_rad))
    new_lon_rad = lon_rad + math.atan2(math.sin(angle_rad) * math.sin(radius / EARTH_RADIUS) * math.cos(lat_rad),
                                        math.cos(radius / EARTH_RADIUS) - math.sin(lat_rad) * math.sin(new_lat_rad))
    return math.degrees(new_lat_rad), math.degrees(new_lon_rad)
    
  def generate_equally_spaced_target_points(self, n=3, radius=CIRCLE_RADIUS):
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

  def create_target_points(self, start_lat, start_lon, radius=CIRCLE_RADIUS, n=3):
    """
    Creates `n` equally spaced target points around a circle centered at the
    provided (start_lat, start_lon), using a given radius in meters.
    """
    circle_points = self.generate_equally_spaced_target_points(n, radius)
    
    target_points = []
    for point in circle_points:
        x, y = point
        angle = np.degrees(np.arctan2(y, x))
        target_lat, target_lon = self.calculate_circle_point(start_lat, start_lon, radius, angle)
        target_points.append((target_lat, target_lon))
    
    return target_points
           
  def parallel_simulation(self, target_points):
    """Runs a parallel simulation using multiple environments and threads."""
    batch_size = self.maxPopulation // NUM_THREADS
    

    with ProcessPoolExecutor(max_workers=NUM_THREADS) as executor:
      futures = []
      
      for i in range(NUM_THREADS):
        #scaler = MinMaxScaler(feature_range=(-1, 1)) 
        batch_start = i * batch_size
        batch_end = batch_start + batch_size
        batch = self.population[batch_start:batch_end]
        population_updated = []
        futures.append(
          executor.submit(evaluate_individuals, batch, self.input_dim, self.output_dim, target_points)
        )
                
      for future in as_completed(futures):
        try:
          batch_results = future.result()
          
          fitness_results = {}
          
          for distance_to_target, crashed, step_count, cumulative_altitude_dist, individual in batch_results:
            current_fitness = self.setFitness(individual, distance_to_target, crashed, step_count, cumulative_altitude_dist)
            population_updated.append(individual)

            if individual not in fitness_results:
              fitness_results[individual] = []

            fitness_results[individual].append(current_fitness)
            
            if len(fitness_results[individual]) == 3:
              avg_fitness = sum(fitness_results[individual]) / 3  
              individual.fitness = avg_fitness
              
            if self.bestIndividual == None or self.bestIndividual.fitness < current_fitness:
              self.bestIndividual = individual

        except Exception as e:
          print(f"Error in parallel simulation: {e}")

      self.population = population_updated
      
      gc.collect()
      print("All episodes completed.")
    
  """Genetic Algorithm Functions"""
  
  def setFitness(self, indiviual, distance, crashed, n_steps, cumulative_altitude_dist):
    """ Sets the fitness of the individual"""
    avg_altitude_dist = cumulative_altitude_dist / n_steps
    crash_penalty = -1000 if crashed else 0
    fitness = (((1 / (distance + 1)) * 1000) / n_steps) * 250 + crash_penalty - avg_altitude_dist
    #print(f'FITNESS: {fitness} || DISTANCE: {distance} || CRASH: {crash_penalty}')
    indiviual.fitness = fitness
    return fitness
      
  def generatePopulation(self): 
    """Creates a number of random genomes and adds them into the population as (genome, fitness) tuples"""
    self.population = [Individual(self.nn.generate_random_genome(self.nn)) for _ in range(self.maxPopulation)]
    id = 1
    for i in self.population:
      i.id=id
      id+=1
      
  def keep_elite(self):
    """Keeps the elite of the genomes for the next generation"""
    elitism_count = int(self.elitismRate * self.maxPopulation)
    new_population = []
    for i in range(0, elitism_count):
      new_population.append(self.population[i])
    return new_population
  
  def tournament_selection(self):
    """Used to select the parents for crossover but in a tournament style"""
    tournament_population = []
    for i in range(0, self.tournamentSize):
      tournament_population.append(random.choice(self.population))
      
    tournament_population.sort(key=lambda Ind: Ind.fitness, reverse=True)
    return tournament_population[0]
  
  def crossover(self, new_population, parent1, parent2):
    """Creates new genomes based on previous genomes of the previous generation"""
    #crossOverPoint = random.randint(0, len(parent1))
    #child1 = np.concatenate((parent1[:crossOverPoint], parent2[crossOverPoint:]))
    #child2 = np.concatenate((parent2[:crossOverPoint], parent1[crossOverPoint:]))

    new_population.append(Individual(parent1))
    new_population.append(Individual(parent2))
    
  def mutate(self, genome):
    """Mutates each new genome made by the crossover feature"""
    for i in range(0, len(genome)):
      mutation_flag = random.random()
      if mutation_flag < self.mutationProb:
        
        factor = random.uniform(-0.1, 0.1)
        genome[i] += factor 
        if genome[i] > 1:
          genome[i] = 1
        if genome[i] < -1:
          genome[i] = -1
  
  def evolve(self):

    with open('fitness_evolution.txt', 'w') as file:
      pass
    file.close()
    target_points = self.create_target_points(START_LAT, START_LON)
    self.generatePopulation()
        
    for i in range(0, self.generationMax):
      self.parallel_simulation(target_points) 
      self.population.sort(key=lambda Ind: Ind.fitness, reverse=True)
        
      self.bestIndividual = self.population[0]
      if self.bestIndividual != None:
        best_model = self.nn.genome_to_model(self.bestIndividual.genome)
        best_model.save("best_model.h5")
        save_logs(self.bestIndividual)
        save_fitness_log(self.bestIndividual, i)
        save_pry(self.bestIndividual)
        save_avg_fitness_log(i, self.population)
        
        #output_folder = "log_files"
        #for episode_number, episode_data in self.bestIndividual.ardupilot_log.items():
        #  output_file = os.path.join(output_folder, f"simulation_{episode_number}.tlog")
        #  observations_to_tlog(episode_data, 1, output_file)
              
      print(f'Generation {i}, Best Fitness: {self.bestIndividual.fitness}')
        
      new_population = self.keep_elite()
        
      old_pop_len = len(self.population)
        
      while len(new_population) < old_pop_len:
        parent1 = self.tournament_selection()
        parent2 = self.tournament_selection()
        self.crossover(new_population, parent1.genome, parent2.genome)
        
      for i in range(int(self.elitismRate * self.maxPopulation), len(new_population)):
        self.mutate(new_population[i].genome)
        
      K.clear_session()
        
      self.population.clear()
      self.population = new_population

      for individual in new_population:
        individual.log.clear()
        individual.pry.clear()
      
      
""" LOGS """
def save_logs(bestIndividual):
  with open("best_individual_log.txt", "w") as log_file:
    
    log_file.write(f"Best Fitness: {bestIndividual.fitness}\n")
    log_file.write(f"Target Latitude: {bestIndividual.log[0].split(',')[0].split(':')[1].strip()}, ")
    log_file.write(f"Target Longitude: {bestIndividual.log[0].split(',')[1].split(':')[1].strip()}, ")
    log_file.write(f"Target Altitude: 300m\n")  
            
    for step_log in bestIndividual.log[1:]:  
      log_file.write(step_log + "\n")  

def save_fitness_log(bestIndividual, gen):
  with open("fitness_evolution.txt", "a") as log_file:
    log_file.write(f"Generation {gen}: {bestIndividual.fitness}\n")

def save_avg_fitness_log(gen, population):
  avg = 0
  for individual in population:
    avg += individual.fitness

  avg /= len(population)

  with open("fitness_avg_evolution.txt", "a") as log_file:
    log_file.write(f"Generation {gen}: {avg}\n")

def save_pry(bestIndividual): #PITCH ROLL YAW
  with open("pitch_roll_yaw_evolution.txt", "w") as log_file:
    for step_log in bestIndividual.pry:
      log_file.write(step_log + "\n")

def validate_data(values):
    """Validates a single observation entry."""
    try:
        assert -3.14159 <= values[0] <= 3.14159, "Roll out of range (-π to π)"
        assert -3.14159 <= values[1] <= 3.14159, "Pitch out of range (-π to π)"
        assert -3.14159 <= values[2] <= 3.14159, "Yaw out of range (-π to π)"
        assert 0 <= values[4] * 1000, "Altitude must be positive"
        assert 0 <= values[8] * 1000, "MSL Altitude must be positive"
        assert -90 <= values[9] <= 90, "Latitude out of range (-90 to 90 degrees)"
        assert -180 <= values[10] <= 180, "Longitude out of range (-180 to 180 degrees)"
        assert 0 <= values[15] <= 36000, "Heading out of range (0 to 360 degrees x100)"
        assert 0 < values[11], "north < 0"
        assert 0 < values[12], "east < 0"
        assert 0 < values[14], "groundspeed < 0"
        return True
    except AssertionError as e:
        print(f"Validation error: {e}")
        return False

def observations_to_tlog(observations, timestep_sec, output_file):
    """
    Converts an episode's observations into a .tlog file.

    Parameters:
        observations (list): A list containing observations for the entire episode.
        timestep_sec (float): Time interval between observations in seconds.
        output_file (str): Path to save the .tlog file.

    Returns:
        None
    """
    with open(output_file, 'wb') as tlog_file:
      mav = mavlink.MAVLink(tlog_file)
      mav.srcSystem = 1  # System ID
      mav.srcComponent = 1  # Component ID

      # Initialize time tracking
      current_time_ms = 0  # Time since start, in milliseconds
      
      step=1
      for values in observations:
        #validate_data(values)

        current_time_ms += int(timestep_sec * 1000)  # Add timestep in milliseconds

        current_roll = values[0]  # Roll in radians
        current_pitch = values[1]  # Pitch in radians
        current_yaw = values[2]  # Yaw in radians
        throttle = values[3] * 100  # Throttle as percentage
        current_altitude = int(values[4] * 1000)  # Altitude (AGL) in millimeters
        current_altitude_msl = int(values[8] * 1000)  # Altitude (MSL) in millimeters
        current_lat = int(values[9] * 1e7)  # Latitude in 1E-7 degrees
        current_lon = int(values[10] * 1e7)  # Longitude in 1E-7 degrees
        velocity_north = int(values[11])  # North velocity in cm/s
        velocity_east = int(values[12])  # East velocity in cm/s
        velocity_down = int(values[13])  # Down velocity in cm/s
        ground_speed = int(values[14])  # Ground speed in cm/s
        heading = int(values[15]) % 36000  # Heading in centi-degrees
        roll_speed = values[16]  # Roll rate in rad/s
        pitch_speed = values[17]  # Pitch rate in rad/s
        yaw_speed = values[18]  # Yaw rate in rad/s

        # Send HEARTBEAT message every 5 steps
        if step % 5 == 0:
            heartbeat_msg = mavlink.MAVLink_heartbeat_message(
                type=mavlink.MAV_TYPE_FIXED_WING,
                autopilot=mavlink.MAV_AUTOPILOT_GENERIC,
                base_mode=0,
                custom_mode=0,
                system_status=mavlink.MAV_STATE_ACTIVE,
                mavlink_version=3
            )
            mav.send(heartbeat_msg)

        # Send GPS_RAW_INT message
        gps_msg = mavlink.MAVLink_gps_raw_int_message(
            time_usec=current_time_ms * 1000,  # Time in microseconds
            fix_type=3,  # 3 = 3D Fix
            lat=current_lat,
            lon=current_lon,
            alt=current_altitude_msl,
            eph=50,  # Horizontal dilution (cm)
            epv=50,  # Vertical dilution (cm)
            vel=ground_speed,  # Ground speed (cm/s)
            cog=heading,  # Course over ground (centidegrees)
            satellites_visible=10  # Number of satellites
        )
        mav.send(gps_msg)

        # Send ATTITUDE message
        attitude_msg = mavlink.MAVLink_attitude_message(
            time_boot_ms=current_time_ms,  # Time in milliseconds since boot
            roll=current_roll,  # Roll in radians
            pitch=current_pitch,  # Pitch in radians
            yaw=current_yaw,  # Yaw in radians
            rollspeed=roll_speed,  # Roll rate (rad/s)
            pitchspeed=pitch_speed,  # Pitch rate (rad/s)
            yawspeed=yaw_speed  # Yaw rate (rad/s)
        )
        mav.send(attitude_msg)

        # Send GLOBAL_POSITION_INT message
        position_msg = mavlink.MAVLink_global_position_int_message(
            time_boot_ms=current_time_ms,  # Time in milliseconds since boot
            lat=current_lat,
            lon=current_lon,
            alt=current_altitude_msl,
            relative_alt=current_altitude,
            vx=velocity_north,
            vy=velocity_east,
            vz=velocity_down,
            hdg=heading
        )
        mav.send(position_msg)
        step+=1
        
      print(f"TLog file successfully created: {os.path.abspath(output_file)}")


if __name__ == "__main__":
  GA = Genetic_Algorithm(8, 4, 100, 1500, 0.1, 5, 0.15)
  GA.evolve()
  