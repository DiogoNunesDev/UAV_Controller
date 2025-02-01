# Autonomous Drone Controller using Deep Reinforcement Learning 🚀

## Overview

This project explores the development of an **autonomous drone controller** using **Deep Reinforcement Learning**. The goal is to train an AI agent to control a UAV in simulated environments, optimizing flight stability and navigation.

Currently, the project leverages:
- **gym-jsbsim** for creating a high-fidelity reinforcement learning environment.
- **Evolutionary algorithms** for initial policy optimization.
- **Actor-Critic Proximal Policy Optimization (PPO)** for training the agent.
- **Visualization tools** to analyze UAV trajectories and sensor data.

---

## 🛠 Features

- **Simulated UAV Environment**: Uses **gym-jsbsim** to simulate real-world UAV physics.
- **Deep Reinforcement Learning**: Implements **Actor-Critic PPO** to optimize flight stability and path planning.
- **Evolutionary Algorithms**: Applied to explore and enhance training mechanisms.
- **Trajectory Visualization**: Tools for analyzing UAV paths, altitude changes, and sensor readings.

---

## 🎮 Simulation & Training Environment

The simulation environment is designed to provide realistic UAV dynamics while allowing deep reinforcement learning models to interact with the system. **gym-jsbsim** acts as the main reinforcement learning interface, offering fine-grained control over aircraft states.

### Why **gym-jsbsim**?
- It integrates the **JSBSim** physics engine with a **Gym** environment.
- Provides higher **simulation fidelity** compared to basic gym environments.
- Allows greater **control over UAV dynamics** for precise learning.
- Supports **flexible reward shaping** to optimize specific flight behaviors.

### Environment Features
- Updated **gym-jsbsim** package to include an initial task of navigating from point A to B.
- A **Target Point (Objective)** is generated at a distance of **500m** and an altitude of **300m**.
- The agent receives inputs from aircraft sensors and outputs attitude changes.
- The first goal is to get as close as possible to the **Objective** while maintaining altitude and reaching the target in the shortest time possible.

---

## 🏗 Model & Algorithm: Actor-Critic PPO

The project employs **Proximal Policy Optimization (PPO)**, an effective on-policy reinforcement learning method that balances exploration and exploitation.

### Why PPO?
- **Stable and efficient** for continuous control problems.
- **Improved convergence** over vanilla policy gradient methods.
- **Adaptable to complex UAV dynamics** with state constraints.

The **Actor-Critic framework** ensures the agent learns optimal control policies by combining:
- **Actor Network**: Learns the optimal flight policy.
- **Critic Network**: Evaluates the quality of actions taken.

---

## 📈 Visualization Tools
To monitor training and analyze flight behavior, we implemented visualization tools for:
- **UAV path trajectories** (2D).
- **Flight stability** and **altitude changes**.
- **Sensor data readings** over time.

---

## 📊 Results & Performance Evaluation
Key metrics used to evaluate the model:
- **Reward Convergence**: Tracks how the PPO agent improves over time.
- **Flight Stability**: Measures altitude deviations and roll stability.
- **Path Efficiency**: Evaluates trajectory deviation from optimal paths.

---

## 🚀 Future Directions
Planned improvements include:
- **Flight log file** to enable 3D visualization in popular tools like: https://www.flightcoach.org/ribbon/plotter.html
- **Integration of Evolutionary Algorithms** for enhanced optimization.
- **Multi-Agent Coordination**: Enabling multiple UAVs to navigate collaboratively.
- **Adaptive Learning**: Refining control policies based on real-time flight data.