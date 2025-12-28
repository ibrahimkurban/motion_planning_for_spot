# Motion Planning for Spot

This repository implements a motion planning and control framework for the Boston Dynamics Spot robot using **Drake**. It focuses on dynamic locomotion over complex, non-convex terrain (stepping stones) using **Graph of Convex Sets (GCS)** for footstep planning and trajectory optimization for full-body control.


### 1. End-to-End Simulation (`demo_main.ipynb`)
The primary entry point that combines terrain generation, planning, and simulation.
* **Pipeline**: Terrain -> GCS Footstep Plan -> Trajectory Generation -> Stabilization.
* **Simulation**: Runs a full physics simulation of Spot traversing the generated terrain.
* **Visualization**: Uses **Meshcat** for real-time 3D visualization of the robot and environment.

### 2. Robust Footstep Planning (`demo_gcs.ipynb`)
Uses **Graph of Convex Sets (GCS)** to solve the combinatorial problem of choosing footstep locations across a randomized field of "stepping stones" (bridges).
* **Terrain Generation**: Procedurally generates stepping stone environments with configurable gaps and bridges.
* **GCS Optimization**: Formulates the footstep planning problem as a shortest-path problem on a graph where edges contain convex constraints (valid kinematic regions).
* **Mosek Solver**: Uses Mosek to efficiently solve the resulting convex optimization problem.

### 3. Trajectory Optimization (`demo_trajopt_self_contained.ipynb`)
A standalone demonstration of low-level trajectory optimization constraints.
* **Constraints**: Implements velocity matching, angular momentum, and friction cone constraints directly within `pydrake`'s `MathematicalProgram`.
* **Solver**: Uses SNOPT/IPOPT for non-linear trajectory optimization.

## 📂 Project Structure

* `src/`
    * `MyGCS.py`: Core logic for the Graph of Convex Sets footstep planner.
    * `MyTerrain.py`: Utilities for generating random stepping-stone terrains.
    * `MyPlanner.py`: Full-body motion planner logic.
    * `MyController.py`: Hybrid controller implementation for tracking trajectories.
* `models/`: URDF and mesh files for Spot and the environment.
* `utils/`: Helper functions for plotting, URDF loading, and trajectory math.


### Prerequisites

* **Python 3.8+**
* **Drake**: [Installation Guide](https://drake.mit.edu/installation.html)
* **Mosek**: Required for the GCS planner. You need a valid license file.

### Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/ibrahimkurban/motion_planning_for_spot.git](https://github.com/ibrahimkurban/motion_planning_for_spot.git)
    cd motion_planning_for_spot
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Important**: Configure your Mosek license.
    The notebooks expect the `MOSEKLM_LICENSE_FILE` environment variable to be set. You can set this in your shell or update the path in the notebooks directly:
    ```python
    import os
    os.environ["MOSEKLM_LICENSE_FILE"] = "/path/to/your/mosek.lic"
    ```

### Running the Demos

1.  Start the Jupyter Notebook server:
    ```bash
    jupyter notebook
    ```

2.  Open **`demo_main.ipynb`** to run the full simulation.
    * Run the first cell to start **Meshcat**.
    * Click the link printed in the output (e.g., `http://localhost:7000`) to view the visualizer.
    * Run subsequent cells to generate the terrain, plan the footsteps, and simulate the motion.


* **Mosek**: Used for the Mixed-Integer Convex Optimization / GCS footstep planning.
* **SNOPT / IPOPT**: Used for non-linear trajectory optimization tasks within Drake.

