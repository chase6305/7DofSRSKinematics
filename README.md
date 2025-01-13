# 7DofSRSKinematics

This repository provides a geometric analytical implementation in Python for the 7DOF KUKA iIWA robot.

## Reference Papers

The main references for this implementation are:

- **Analytical Inverse Kinematic Computation for 7-DOF Redundant Manipulators With Joint Limits and Its Application to Redundancy Resolution**
  - *Masayuki Shimizu, Hiromu Kakuya, Woo-Keun Yoon, Kosei Kitagaki, Kazuhiro Kosuge*
  - IEEE Transactions on Robotics, 2008
  - DOI: [10.1109/TRO.2008.2003266](https://doi.org/10.1109/TRO.2008.2003266)

- **Position-based kinematics for 7-DoF serial manipulators with global configuration control, joint limit, and singularity avoidance**
  - *Carlos Faria, Flora Ferreira, Wolfram Erlhagen, Sérgio Monteiro, Estela Bicho*
  - Mechanism and Machine Theory, 2018
  - DOI: [10.1016/j.mechmachtheory.2017.10.025](https://doi.org/10.1016/j.mechmachtheory.2017.10.025)

## Summary

This repository implements an analytical method to solve the inverse kinematics of 7-DOF redundant manipulators while avoiding joint limits and singularities. The method introduces two auxiliary parameters to manage self-motion manifolds: the global configuration (GC) and the arm angle (ψ). The global configuration specifies the branch of inverse kinematics solutions, while the arm angle parameterizes the elbow redundancy within the specified branch.

### Key Features

- **Global Configuration Control**: Determines the branch of inverse kinematics solutions.
- **Joint Limit Avoidance**: Maps joint limits to arm angle values to determine feasible intervals.
- **Singularity Avoidance**: Avoids singularities by calculating joint angles from the position-based inverse kinematics algorithm.
- **Real-time Implementation**: Suitable for real-time control systems without the disadvantages of using the Jacobian matrix.
- **Redundancy Resolution**: Solves both global and local manifolds using a redundancy resolution strategy.

### Implementation

The repository includes a Python implementation of the described methods, enabling real-time control and redundancy resolution for 7DOF KUKA-iiwa robots.
