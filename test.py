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


STEP_FREQUENCY_HZ = 5  # Frequency at which actions are sent
EPISODE_TIME_S = 60  # Total episode duration in seconds
RESET_AFTER_EPISODES = 3

def create_env():
    
    env = JsbSimEnv(
        task_type=NavigationTask,
        aircraft=cessna172P,
        agent_interaction_freq=STEP_FREQUENCY_HZ,
        shaping=None,  
    )

    """
    properties_to_display = [
        prp.altitude_agl_ft, prp.roll_rad, prp.pitch_rad,
        prp.heading_deg, prp.throttle, prp.aileron_cmd,
        prp.elevator_cmd, prp.rudder_cmd
    ]
    """
    return env


def run_random_controller():
    env = create_env()
    episode_count = 0    

    while episode_count < RESET_AFTER_EPISODES:
        obs = env.reset()
        env.render()
        done = False
        step_count = 0
        
        print(f"Starting episode {episode_count + 1}")

        # Main loop for each episode
        while not done:
            
            action = np.array([0.0, 0.0, 0.0, 1.0])  # Stable Roll, Pitch, Yaw, Throttle

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