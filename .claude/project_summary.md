# Muscle Imitation Learning Study - Project Summary

## Overview
A Python-based muscle simulation and deep reinforcement learning framework for creating anatomically accurate muscle models for physics-based character simulation. This project enables learning muscle-driven locomotion through imitation learning.

## Core Purpose
- Train neural networks to control musculoskeletal models that imitate reference motions (e.g., walking, running)
- Simulate realistic muscle-tendon dynamics for physics-based character animation
- Retarget muscle models between different skeletal structures (e.g., SMPL to custom skeletons)

## Architecture

### Main Entry Points
- `main.py` - Launches the GLFW-based OpenGL viewer for visualization and interaction
- `train.py` - Runs distributed PPO training using Ray RLlib

### Key Modules

#### `/core/`
- `env.py` - OpenAI Gym-compatible environment wrapping DART physics simulation
  - Manages skeleton, muscles, ground plane
  - Handles BVH motion loading for reference poses
  - Implements reward functions (COM tracking, end-effector matching)
  - Supports multiple actuator types: "pd_ref_residual", "mass" (muscle-actuated)
- `dartHelper.py` - Utilities for building DART skeletons from XML
- `bvhparser.py` - BVH motion file parser
- `smplparser.py` - SMPL body model parser

#### `/learning/`
- `ray_model.py` - Neural network architectures:
  - `SimulationNN` - Policy network (actor-critic) with configurable hidden layers
  - `MuscleNN` - Maps desired torques to muscle activations (supervised learning)
  - `PolicyNN` - Wrapper for inference with observation filters
- `ray_ppo.py` - Custom PPO trainer extending Ray RLlib
- `ray_torch_policy.py` - Custom GAE computation for trajectory postprocessing
- `ray_config.py` - Training hyperparameter configurations

#### `/viewer/`
- `viewer.py` - Main GLFW/OpenGL application (imgui-based GUI)
  - 3D visualization of skeleton, muscles, and meshes
  - Interactive camera controls (trackball rotation)
  - Real-time simulation playback
  - Muscle activation visualization
- `TrackBall.py` - Quaternion-based camera trackball
- `gl_function.py` - OpenGL drawing utilities
- `mesh_loader.py`, `muscle_mesh.py`, `skeleton_mesh.py` - Mesh handling
- `fiber_architecture.py` - Muscle fiber visualization
- `arap_backends.py` - As-Rigid-As-Possible mesh deformation (GPU/Taichi support)

#### `/utils/`
- `pose_tracker.py` - OpenPose/STAF integration for video-to-pose

### Data & Assets

#### `/data/`
- `env.xml`, `env_skel.xml` - Environment configuration (skeleton, ground, motion files)
- Skeleton definitions and muscle attachment files

#### `/Zygote_Meshes*/`
- High-resolution anatomical meshes (muscles, bones) from Zygote 3D anatomy
- Multiple versions: original, revised, subdivided

#### `/skel/`
- Skeleton definition files

### Configuration Files
- `calc.py` - Joint position mappings between SMPL (24 joints) and custom skeleton (23 joints)
- `skeleton_section.py` - DART skeleton info dictionary

## Key Technologies
- **Physics Engine**: DART (dartpy) - Rigid body dynamics with muscle simulation
- **Deep RL**: Ray RLlib 2.0.1 with PPO algorithm
- **Neural Networks**: PyTorch 2.0.1
- **Visualization**: OpenGL (PyOpenGL), GLFW, imgui
- **Mesh Processing**: trimesh, scipy

## Muscle Model
The muscle simulation uses Hill-type muscle-tendon units with parameters:
- `f0` - Maximum isometric force
- `lm` - Optimal muscle fiber length
- `lt` - Tendon slack length
- `pen_angle` - Pennation angle
- `lmax` - Maximum muscle length

Muscles are defined as waypoints attached to body segments, with support for:
- Single fiber muscles
- Multi-fiber muscles (Zygote format)
- Weight-based attachment for mesh deformation

## Training Pipeline
1. Load environment from XML (skeleton + muscles + reference motion)
2. Initialize Ray cluster (local or distributed)
3. PPO trains policy network to minimize tracking error
4. Muscle network learns activation patterns via supervised learning
5. Checkpoints saved periodically for best/latest models

## Reinforcement Learning Details
- **Observation**: Body node positions, velocities, orientations (root-relative) + target poses
- **Action**: Joint displacement deltas (scaled by `action_scale`)
- **Reward**: Exponential decay based on COM error, end-effector error
- **Episode termination**: Low reward threshold or time limit (10s)

## Usage
```bash
# Visualization
python main.py --env_path data/env_skel.xml

# Training
python train.py --config=ppo_small_pc --env=data/env.xml --name=experiment_name

# Load trained model
python main.py --checkpoint path/to/checkpoint
```

## Python Version & Dependencies
- Python 3.8
- DART 6.11.1 (custom dartpy build with muscle extensions)
- Ray 2.0.1, PyTorch 2.0.1
- PyOpenGL, imgui, glfw, numpy, numba, trimesh
