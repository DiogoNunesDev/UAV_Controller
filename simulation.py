from neural_network import NeuralNetwork
import random
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
import numpy as np
from gym_jsbsim.environment import JsbSimEnv
from gym_jsbsim.tasks import NavigationTask  
from gym_jsbsim.aircraft import cessna172P
import gym_jsbsim.properties as prp
import gc
from tensorflow.keras import backend as K

#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)




STEP_FREQUENCY_HZ = 5  # Frequency at which actions are sent
EPISODE_TIME_S = 10  # Total episode duration in seconds
EARTH_RADIUS = 6371000  # Earth radius in meters
CIRCLE_RADIUS = 300     # Circle radius in meters
NUM_POINTS = 15         # Number of points on the circumference
TOLERANCE_DISTANCE = 10 # Tolerance distance in meters
ALTITUDE_THRESHOLD = 100  # Altitude threshold to detect crash or failure
NUM_THREADS = 3

class Individual:
  def __init__(self, genome):
    self.genome = genome
    self.fitness = None
    self.log = []
    
    
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
  
def create_env():
  """Sets up the GymJSBSim environment for the navigation task."""
  env = JsbSimEnv(
    task_type=NavigationTask,
    aircraft=cessna172P,
    agent_interaction_freq=STEP_FREQUENCY_HZ,
    shaping=None,
  )
  return env
  
def evaluate_individuals(individuals, input_dim, output_dim):
  """Evaluates a batch of individuals in the provided environment instance."""
  
  env = create_env()
  
  results = []
  for individual in individuals:
    model = NeuralNetwork(input_dim, output_dim).genome_to_model(individual.genome)
    obs = env.reset()
    done, step_count, total_time, crashed = False, 0, 0, False

    current_log = []
    individual.log.append(f"Target Latitude: {env.task.target_point[0]:.6f}, Target Longitude: {env.task.target_point[1]:.6f}, Target Altitude: 300m")
    individual.log.append("Step\tLatitude\tLongitude\tAltitude")
    
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
      
      individual.log.append(f"{step_count}\t{current_lat:.6f}\t{current_lon:.6f}\t{current_alt * 0.3048}")

    distance_to_target = info.get('distance_to_target', float('inf'))
    results.append((distance_to_target, total_time, crashed, step_count, current_alt, individual))

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
    #self.envs = [self.create_env() for _ in range(NUM_THREADS)]
    
     

           
  def parallel_simulation(self, gen):
    """Runs a parallel simulation using multiple environments and threads."""
    batch_size = self.maxPopulation // NUM_THREADS
    with ProcessPoolExecutor(max_workers=NUM_THREADS) as executor:
      futures = []
      
      for i in range(NUM_THREADS):
        batch_start = i * batch_size
        batch_end = batch_start + batch_size
        batch = self.population[batch_start:batch_end]
        #env = self.envs[i]
        
        futures.append(
          executor.submit(evaluate_individuals, batch, self.input_dim, self.output_dim)
        )
                
      for future in as_completed(futures):
        batch_results = future.result()
        
        for distance_to_target, total_time, crashed, step_count, current_alt, individual in batch_results:
          current_fitness = self.setFitness(individual, distance_to_target, total_time, crashed, step_count, current_alt)
          
          if individual.fitness == None:
            print("NONENONENONE")
            
          if self.bestIndividual == None or self.bestIndividual.fitness < current_fitness:
            self.bestIndividual = individual

      
      gc.collect()
      print("All episodes completed.")
  
  
  
  """OLD FUNCTION"""
  def simulation(self, gen):
    """Runs a simulation episode using GymJSBSim and the neural network for control."""
    episode_count = 1
    max_steps = EPISODE_TIME_S * STEP_FREQUENCY_HZ
    best_individual_path = "best_individual_log.txt"
    current_log = []  

    for individual in self.population:
      print(f'Generation: {gen} Individual {episode_count}')
      model = self.nn.genome_to_model(individual.genome)
        
      obs = self.env.reset()
      done = False
      step_count = 0
      total_time = 0
      crashed = False
        
      target_lat = self.env.task.target_point[0]
      target_lon = self.env.task.target_point[1]

      # Clear the log for the current individual
      current_log.clear()
      current_log.append(f"Target Latitude: {target_lat:.6f}, Target Longitude: {target_lon:.6f}, Target Altitude: 300m")
      current_log.append("Step\tLatitude\tLongitude\tAltitude")

      while not done and step_count < max_steps:
        input_vector = np.array(obs).reshape(1, -1)
        action = model.predict(input_vector)[0]
                
        roll, pitch, yaw = action[:3]
        throttle = (action[3] + 1) / 2
        action = np.array([roll, pitch, yaw, throttle])
                
        obs, reward, done, info = self.env.step(action)
        step_count += 1
        total_time += 1 / STEP_FREQUENCY_HZ
                
        current_lat = self.env.sim[prp.lat_geod_deg]
        current_lon = self.env.sim[prp.lng_geoc_deg]
        current_alt = self.env.sim[prp.altitude_agl_ft]
        crashed = (current_alt * 0.3048) <= ALTITUDE_THRESHOLD

        current_log.append(f"{step_count}\t{current_lat:.6f}\t{current_lon:.6f}\t{current_alt * 0.3048}")

        
      distance_to_target = info.get('distance_to_target', float('inf'))
      current_fitness = self.setFitness(individual, distance_to_target, total_time, crashed, step_count, current_alt)
      
                                   
      if self.bestIndividual is None or current_fitness > self.bestIndividual.fitness:
        self.bestIndividual = individual
        with open(best_individual_path, "w") as best_log: 
          best_log.write(f"Best Fitness: {self.bestIndividual.fitness}\n")
          best_log.write("\n".join(current_log))

      episode_count += 1

    gc.collect()
    print("All episodes completed.")

  
  """Genetic Algorithm Functions"""
  
  def setFitness(self, indiviual, distance, total_time, crashed, n_steps, final_altitude):
    """ Sets the fitness of the individual"""
    crash_penalty = -1000 if crashed else 0
    fitness = ((1 / (distance + 1)) * 1000) / n_steps + crash_penalty - (1/(abs(300 - final_altitude) + 1)) * 100
    #print(f'FITNESS: {fitness} || DISTANCE: {distance} || CRASH: {crash_penalty}')
    indiviual.fitness = fitness
    return fitness
      
  def generatePopulation(self): 
    """Creates a number of random genomes and adds them into the population as (genome, fitness) tuples"""
    self.population = [Individual(self.nn.generate_random_genome(self.nn)) for _ in range(self.maxPopulation)]
      
  def keep_elite(self):
    """Keeps the elite of the genomes for the enxt generation"""
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
    crossOverPoint = random.randint(0, len(parent1))
    child1 = np.concatenate((parent1[:crossOverPoint], parent2[crossOverPoint:]))
    child2 = np.concatenate((parent2[:crossOverPoint], parent1[crossOverPoint:]))

    new_population.append(Individual(child1))
    new_population.append(Individual(child2))
    
  def mutate(self, genome):
    """Mutates each new genome made by the crossover feature"""
    for i in range(0, len(genome)):
      mutation_flag = random.random()
      if mutation_flag < self.mutationProb:
        genome[i] = random.uniform(-1, 1)
        
  def evolve(self):
    self.generatePopulation()
        
    for i in range(0, self.generationMax):
      self.parallel_simulation(i) 
      self.population.sort(key=lambda Ind: Ind.fitness, reverse=True)
        
      self.bestIndividual = self.population[0]
              
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
      
    for env in self.envs:
      env.close()

      

if __name__ == "__main__":
  
  GA = Genetic_Algorithm(8, 4, 300, 500, 0.1, 5, 0.1)
  GA.evolve()
  