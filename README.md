# Autonomous Drone Controller using Deep Reinforcement Learning 🚀

## Overview

This project explores the development of an **autonomous drone controller** using **Deep Reinforcement Learning**. The goal is to train an AI agent to control a UAV in simulated environments, optimizing flight stability and navigation.

This project implements a custom **NavigationTask** on top of **gym-jsbsim** to teach a fixed-wing agent to:
- Fly from a known start (pointing North) to a generated **target point**.
- Maintain **300 m AGL** while turning/aligning to the target.
- Finish within a **success radius** and altitude band, under attitude limits.

---

## Features

- **Simulated UAV Physics Environment**: Uses **gym-jsbsim** to simulate real-world UAV physics.
- **Deep Reinforcement Learning**: Implements **Actor-Critic PPO** to optimize flight stability and path planning.
- **Curriculum Learning**: structured progression from straight flight to harder turns
- **Trajectory Visualization**: Tools for analyzing UAV paths, altitude changes, and sensor readings.

---

## Simulation & Training Environment

The simulation environment is designed to provide realistic UAV dynamics while allowing deep reinforcement learning models to interact with the system. **gym-jsbsim** acts as the main reinforcement learning interface, offering fine-grained control over aircraft states.

### Environment Features
- Updated **gym-jsbsim** package to include an initial task of navigating from point A to B.
- A **Target Point (Objective)** is generated according to the curriculum learning setup.
- The agent receives inputs from aircraft sensors and outputs attitude changes.
- The first goal is to get as close as possible to the **Objective** while maintaining altitude, good attitude behaviours and reaching the target in the shortest time possible.

#### Curriculum Learning

The `NavigationTask` defines a sequence of **levels (L0–L5)** with progressively harder objectives:

| Level | Bearing (°) | Distance (m)       | Step (m) | Hits | Success radius (m) | Alt band (m) |
|------:|-------------|--------------------|----------|------|---------------------|--------------|
| L0    | 0           | 500–1000           | 250      | 10   | 30                  | 60           |
| L1    | 0           | 500–2000           | 250      | 10   | 30                  | 60           |
| L2    | ±25         | 500–2000           | 250      | 5    | 30                  | 50           |
| L3    | ±25         | 500–2000           | 250      | 5    | 25                  | 40           |
| L4    | ±50         | 1000–3000          | 500      | 5    | 25                  | 40           |
| L5    | ±90         | 1500–4000          | 500      | 5    | 20                  | 30           |

**Tokens (episode types):**
- **F** → forward (used in early levels)  
- **D** → right turn  
- **E** → left turn  
- **R** → rehearsal (sampling from prior levels)

The curriculum advances distance or promotes to the next level once required **hits** are achieved on each side (if it has bearing). Fail streaks are tracked to prevent premature promotion.
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
- **Creation of a unified autonomous control system** that integrates multiple reinforcement learning (RL)-based tasks, allowing the aircraft to perform complex maneuvers with a single click.
