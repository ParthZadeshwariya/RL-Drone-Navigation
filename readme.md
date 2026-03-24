# Autonomous 3D Drone Navigation using Reinforcement Learning

## Project Overview
This project focuses on developing an autonomous navigation system for a quadrotor drone in a complex 3D environment. Using **Deep Reinforcement Learning (DRL)**, specifically the **Proximal Policy Optimization (PPO)** algorithm, the drone learns to navigate from a starting point to a target goal while successfully avoiding various obstacles like buildings, trees, and rocks.

---

## 1. Environment Design (`drone_env.py`)
The heart of the project is a custom **Gymnasium** environment that simulates realistic drone flight physics and sensor data.

### World Specifications
- **Dimensions**: A continuous 3D workspace of 50m × 50m × 20m.
- **Drone Physics**: The drone operates based on velocity commands on three axes (X, Y, Z).
- **Obstacle Generation**: Every episode generates a random assortment of:
  - **Buildings**: Axis-aligned boxes of varying heights.
  - **Trees**: Vertical cylinders with spherical canopies.
  - **Rocks**: Spherical boulders scattered on the ground.

### Agent Configuration
- **Action Space**: Continuous 3D velocity vector `[vx, vy, vz]` in the range `[-1, 1]`, scaled to a maximum speed of 1.5 m/step.
- **Observation Space (20 dimensions)**:
  - **Goal Vector (3)**: Unit direction towards the goal.
  - **Goal Distance (1)**: Normalized distance to the target.
  - **Position (3)**: Normalized current coordinates.
  - **Velocity (3)**: Current flight velocity.
  - **LiDAR (10)**: 10-directional ray casting for obstacle detection (Forward, Backward, Left, Right, Up, Down, and 4 horizontal diagonals).

---

## 2. Training Pipeline (`train.py`)
The agent is trained using the **Stable Baselines 3** library.

### Key Training Features
- **Algorithm**: Proximal Policy Optimization (PPO), chosen for its stability and efficiency in continuous action spaces.
- **Normalization**: Uses `VecNormalize` to scale observations and rewards, which is critical for consistent gradient updates.
- **Parallelization**: 8 parallel environments are used simultaneously to speed up data collection.
- **Network Architecture**: A Multi-Layer Perceptron (MLP) with three hidden layers `[256, 256, 128]`.
- **Learning Schedule**: Trained for **2,000,000 timesteps** with periodic evaluation to save the best-performing model.

---

## 3. High-Fidelity Visualization (`visualize.py`)
To validate the agent's behavior, a custom visualizer was built using the **PyBullet** physics engine.

### Visual Features
- **Detailed Drone Model**: Includes a central body, crossed arms, landing legs, and **four spinning rotors** that animate based on flight status.
- **Procedural Scenery**: Buildings are rendered with windows and rooftops, trees have trunks and canopies, and the ground is lush green.
- **Follow Camera**: A smooth camera system that tracks the drone's position from a tailing perspective.
- **Flight Trail**: A cyan polyline trail that follows the drone, allowing the user to see the path taken.
- **Heads-Up Display (HUD)**: Real-time overlay showing the current episode, distance to the goal, and status alerts (e.g., "GOAL REACHED!", "COLLISION").

---

## 4. Evaluation and Performance (`evaluate.py`)
The model is rigorously tested over **200 episodes** to ensure robustness.

### Metrics Tracked
- **Success Rate**: The percentage of episodes where the drone successfully reaches the goal radius.
- **Collision Rate**: Frequency of impacts with the ground or obstacles.
- **Timeout Rate**: Episodes that exceed the 600-step limit.
- **Efficiency**: Average number of steps taken to reach the goal.

---

## 5. Technical Requirements
- **Python 3.10+**
- **Gymnasium**: Environment interface.
- **Stable Baselines 3**: RL algorithm implementation.
- **PyBullet**: 3D rendering and physics.
- **NumPy & Matplotlib**: Data processing and plotting.

## Conclusion
This project demonstrates the power of reinforcement learning in solving complex 3D navigation tasks. By combining a custom sensor-rich environment with a robust RL algorithm and high-quality visualization, we have created an agent that can reliably navigate through cluttered spaces entirely on its own.
