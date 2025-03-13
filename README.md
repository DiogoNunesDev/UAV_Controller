# Autonomous Drone Controller using Deep Reinforcement Learning 🚀

## Overview

This project explores the development of an **autonomous drone controller** using **Deep Reinforcement Learning**. The goal is to train an AI agent to control a UAV in simulated environments, optimizing flight stability and navigation.

Currently, the project leverages:
- **gym-jsbsim** to create a high-fidelity reinforcement learning environment.
- **Evolutionary algorithms** used for a better understanding of the environment mechanics and a stable setup.
- **Actor-Critic Proximal Policy Optimization (PPO)** to train the agent.
- **Visualization tools** to analyze UAV trajectories and sensor data.

---

## Features

- **Simulated UAV Physics Environment**: Uses **gym-jsbsim** to simulate real-world UAV physics.
- **Deep Reinforcement Learning**: Implements **Actor-Critic PPO** to optimize flight stability and path planning.
- **Evolutionary Algorithms**: Applied to explore and enhance training mechanisms.
- **Trajectory Visualization**: Tools for analyzing UAV paths, altitude changes, and sensor readings.

---

## Simulation & Training Environment

The simulation environment is designed to provide realistic UAV dynamics while allowing deep reinforcement learning models to interact with the system. **gym-jsbsim** acts as the main reinforcement learning interface, offering fine-grained control over aircraft states.

### Environment Features
- Updated **gym-jsbsim** package to include an initial task of navigating from point A to B.
- A **Target Point (Objective)** is generated at a distance between **4000m** and **4500m**,and an altitude of **300m**.
- The agent receives inputs from aircraft sensors and outputs attitude changes.
- The first goal is to get as close as possible to the **Objective** while maintaining altitude and reaching the target in the shortest time possible.

---

## Model & Algorithm: Actor-Critic PPO

The project employs **Proximal Policy Optimization (PPO)**, an effective on-policy reinforcement learning method that balances exploration and exploitation.

![Proximal Policy Optimization](https://github.com/DiogoNunesDev/AirplaneController/blob/main/PPO.png)

---

## Visualization Tools
To monitor training and analyze flight behavior, we implemented visualization tools for:
- **UAV path trajectories** (2D & 3D).
- **Flight stability** and **altitude changes**.
- **Sensor data readings** over time.

---

## Current Progress

Below is the latest progress of our autonomous UAV controller:

### UAV Flight Path 
<p align="center">
  <img src="https://github.com/DiogoNunesDev/AirplaneController/blob/main/Dinamic_Aircraft_Path.gif" width="800px" height="500px" style="display: inline-block; vertical-align: middle; margin-right: 10px;"/>
</p>

### Flight Path & Altitude Analysis in 3D
<p align="center">
  <img src="https://github.com/DiogoNunesDev/AirplaneController/blob/main/3D_Aircraft_Path.png" width="800px" height="500px" style="display: inline-block; vertical-align: middle; margin-right: 10px;"/>
</p>

### Data Analysis
<p align="center">
  <img src="https://github.com/DiogoNunesDev/AirplaneController/blob/main/Sensors.png" width="800px" height="500px" style="display: inline-block; vertical-align: middle; margin-right: 10px;"/>
</p>

---

## Future Directions
Planned improvements include:
- **Flight log file** to enable 3D visualization in popular tools like: https://www.flightcoach.org/ribbon/plotter.html
- **Integration of Evolutionary Algorithms** for enhanced optimization.
- **Multi-Agent Coordination**: Enabling multiple UAVs to navigate collaboratively.
- **Adaptive Learning**: Refining control policies based on real-time flight data.