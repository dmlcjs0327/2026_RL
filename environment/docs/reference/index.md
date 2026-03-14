[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Aerial Gym Simulator

Welcome to the documentation of the Aerial Gym Simulator.

The Aerial Gym Simulator is a high-fidelity physics-based simulator for training Micro Aerial Vehicle (MAV) platforms such as multirotors to learn to fly and navigate cluttered environments using learning-based methods. The environments are built upon the underlying [NVIDIA Isaac Gym](https://developer.nvidia.com/isaac-gym) simulator. It offers aerial robot models for standard planar quadrotor platforms, as well as fully-actuated platforms and multirotors with arbitrary configurations. These configurations are supported with low-level and high-level geometric controllers that reside on the GPU and provide parallelization for the simultaneous control of thousands of multirotors.

This release includes task definition and environment configuration for fine-grained customization of all the environment entities, and a custom rendering framework for depth and segmentation images and custom sensors such as LiDARs. The simulator is open-source and is released under the [BSD-3-Clause License](https://opensource.org/licenses/BSD-3-Clause).

![Aerial Gym Simulator](./gifs/Aerial%20Gym%20Position%20Control.gif)

![RL for Navigation](./gifs/rl_for_navigation_example.gif)

![Depth Frames 1](./gifs/camera_depth_frames.gif) ![Lidar Depth Frames 1](./gifs/lidar_depth_frames.gif)

![Seg Frames 1](./gifs/camera_seg_frames.gif) ![Lidar Seg Frames 1](./gifs/lidar_seg_frames.gif)


## Features

- **Modular and Extendable Design** — Custom environments, robots, sensors, tasks, and controllers; parameters adjustable at runtime. See [Simulation Components](./4_simulation_components.md) and [Customization](./5_customization.md).
- **High-Fidelity Physics Engine** — [NVIDIA Isaac Gym](https://developer.nvidia.com/isaac-gym/download) for multirotor simulation.
- **Parallelized Geometric Controllers** on the GPU for [simultaneous control of many multirotors](./3_robots_and_controllers.md/#controllers).
- **Custom Rendering Framework** ([NVIDIA Warp](https://nvidia.github.io/warp/)) for [custom sensors](./8_sensors_and_rendering.md/#warp-sensors), depth, and segmentation.
- **RL-based control and navigation** — [Scripts and examples for training](./6_rl_training.md).


!!! warning "**Support for Isaac Lab**"
      Support for [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) and [Isaac Sim](https://developer.nvidia.com/isaac/sim) is currently under development.


## Why Aerial Gym Simulator?

The simulator is designed to simulate thousands of MAVs simultaneously with low- and high-level controllers. Custom ray-casting allows fast rendering for depth and segmentation. Training for motor-command policies can be done in under a minute and vision-based navigation policies in under an hour. Examples are provided to get started with custom robots quickly.


## Quick Links

- [Installation](./2_getting_started.md/#installation)
- [Robots and Controllers](./3_robots_and_controllers.md)
- [Sensors and Rendering Capabilities](./8_sensors_and_rendering.md)
- [RL Training](./6_rl_training.md)
- [Simulation Components](./4_simulation_components.md)
- [Customization](./5_customization.md)
- [Sim2Real Deployment](./9_sim2real.md)
- [FAQs and Troubleshooting](./7_FAQ_and_troubleshooting.md)
