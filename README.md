# MarLag project: Marine Lagrangian Simulation Engine

**MarLag** is a high-performance, vectorized Python library designed for 3D/2D Lagrangian particle tracking and larval connectivity modeling in marine environments.
By leveraging **SciPy's adaptive ODE solvers (`solve_ivp`)** and **NumPy vectorization**, MarLag tracks floating particles (e.g., fish or invertebrate larvae) subjected to ocean currents, horizontal diffusion, and biological behavioral constraints.

## Contents
- [Project structure](#project-structure)
- [Key features](#key-features)
- [Requirements](#requirements)
- [Quick Start](#quick-start)

# Project structure
The project is organised as a modular Python (3.13.5) package:
```text
MarLag/
├──data/                          # Simulation configuration file, grids, current fields
├──lagrlib/                       # Core library
    ├──lagrangian.py              # Wrapper script for particle tracking simulation
    ├──larvae_generator.py        # Wrapper script for larvae generator script
    ├──myio                       # Input/output utility sub-package
        ├──spawnign_reader.py     # Parser for particles/spawning input files
        ├──velocity_fields.py     # NetCDF reader and storing object
        └──writer.py              # Exporter for trajectories and final state datasets
    ├──particles                  # Particles state and biology sub-package
        ├──generator.py           # Particles generation routines
        ├──larvae.py              # Larval attributes, aging, and PLD logic
        └──spawning_models.py     # Spatial/temporal spawning distribution models
    ├──physics                    # Displacement sub-package
        ├──engine.py              # ODE rhs function
        └──movement.py            # Movements kernels
    └──utils                      # Utilities sub-package
        ├──multi.py               # Multiprocessing functions (not used)
        └──time.py                # Management of simulation time and calendar
├──json_gen.py                    # Helper script to save anjson files
├──lagrangian_launcher.py         # Execution script to launch particle tracking simulations
└──larvae_gen_launcher.py         # Execution script to launch particles/larvae generation
```
# Key features
1. NumPy Vectorization: Particle state (coordinate X, Y, Z, age) is managed through flattened NumPy arrays. Deaths, strandings, and releases are vectorised, avoiding for cycles.
2. Adaptive SciPy integration: scipy.integrate.solve_ivp resolves the displacement of particles allowing adaptive time-steps.
3. Advanced larval biology: individual variability of Pelagic Larval Duration (PLD) and spawning timing are modelled through tailored funcitons. Real-time monitoring of larval age.
4. Forward and backward tracking: supports both forward dispersion simulations (larval connectivity) and backward trajectory tracking to locate spawning origins.
5. Multiple movement support: modular kinematic kernels representing 2D/3D Advection (including vertical velocity $W$, when available), sub-grid turbulent diffusion via random walk formulations adjusted for earth coordinates, behavioural traits such as diel vertical migration (DVM).
6. Multi-model support: ready-to-use readers for ingestion of gridded daily/monthly current velocity NetCDF fields from CMEMS, POLCOMS, and CMCC climate projection models.

# Requirements
The project makes use of the following Python common libraries:
- numpy
- scipy
- pandas
- xarray
- global_land_mask
- dateutil
- tqdm
- multiprocessing
```
pip install numpy scipy pandas xarray global_land_mask python-dateutil tqdm multiprocessing
```
The library was tested on Python v3.13.5

# Quick Start
## What is a Lagrangian Simulation?
Instead of observing ocean currents from fixed locations (Eulerian approach), a Lagrangian simulation tracks individual particles (e.g., larvae) as independent agents. Each particle's 3D position $(x, y, z)$ is integrated over time, incorporating currents (advection), turbulence (diffusion), and active biological behaviors.
## Spatial Grid
To initiate tracking, MarLag discretizes marine spawning habitats using a regular grid of square or rectangular cells ($\Delta \phi \times \Delta \lambda$). This grid serves two key purposes:
- Discretisation of spawning limit: each cell represents a localised spawning source that releases a specific cohort of particles on designated days.
- Model Alignment: a structured grid aligns particle initialisation with gridded hydrodynamic datasets (CMEMS, POLCOMS, CMCC), preventing accidental land generation.
## Run the script
```larvae_gen_launcher.py``` and ```lagrangian_launcher.py``` ar the two executing script to generate larvae and launch the proper particle tracking simulation. Each script contains an example of options file (.json) that is needed to build the objcets and start the simulaitons.
### Options file
```
{
    "spawning grid" : "data/sim_grid20800.csv", 
    "grid margins" : {
        "left":12.,
        "right":23.5,
        "top":45.8,
        "bottom":34.9,
    },  
    "depth margins" : (0,300),
    "depth range min": 20,
    "depth range max": 250,
    "total particles" : 1000000,
    "PLD": 54,
    "PLD var": 5,
    "spawning season median" : {"17": [[1, 15], [6, 9]], "18": [[12, 22], [8, 6]]}, # original [[2,5], [6,30]] - [[1,12], [8,27]]
    "spawning season duration" : {"17": [90, 105], "18": [67, 67]},
    "field path" :  "data/currents/",
    "check field path" : "/data01/mediterraneo/2020/05/",
    "spawning export path": "results/larvae_gen/",
    "spawning points path": "results/larvae_gen/LG_2026-06-04_FW/",
    "lagrangian export path": "results/lagrangian/",
}
```
after the file is saved as .json, it can be loaded into the two scripts
### larvae_gen_launcher
```
from lagrlib.larvae_generator import LarvaeGenerator

spwn = LarvaeGenerator(
    starting_date = (2020, 12, 1),
    duration = (0, 0, 10),
    backward = False,
    model = "CMCC",
    workers=1,
    item = "n_km2",
    case = "HKE_17_18_19",
    file='data/sim_options.json',
    config = {
        "verbose": True,
        "progress_bar": True,
    },
)

spwn.generate_larvae(export_path = "results/larvae_gen/")
```
Once the ```spwn``` object is initialised with the starting date ```(year, month, day)``` and the sim duration ```(year, month, day)```, you can fine-tune its behavior using the following optional arguments:
- ```backward``` (bool): Controls the temporal direction of the simulation (forward or backward-in-time tracking)
- ```model``` ['CMCC', 'CMEMS', 'POLCOMS']: Specifies the hydrodynamic ocean model
- ```workers``` (int): egulates the number of parallel CPU processes to allocate for spawning cohort generation
- ```item``` (str): Selects the specific column in the spawning grid file indicating the spawning density for each grid cell
- ```case``` (str): Selects the specific biological spawning strategy or scenario
- ``` file``` (str): Defines the path to your simulation options JSON file
- ```config```: Dictionary regulating helper settings:
    - verbose (bool): Toggles terminal execution status messages
    - progress_bar (bool): Toggles the progress bar visualization during run time

### lagrangian_launcher

```
from lagrlib.lagrangian import Lagrangian

lagr = Lagrangian(
    starting_date = (2020, 12, 1),
    duration = (0, 0, 10),
    backward = False,
    model = "CMCC",
    RCP = 45,
    multi = False, # non è implementato al momento
    chunck_n = 1, # non è implementato al momento
    file='data/sim_options.json',
    config = {
        "verbose": True,
        "progress_bar": True,
        "horizontal advection": True,
        "vertical advection": False,
        "diffusion": False,
        "dvm": False,
        "diffusion coefficient": 10.0,
        "diffusion dt": 86400,
        "day depth": 100,
        "night depth": 5,
        "dvm speed": 1.0,
    },
)

lagr.run_sequential()
```
Once you initialize the tracking engine ```lagr``` with a starting date ```(year, month, day)``` and a simulation duration ```(year, month, day)```, you can customize the physics, kinematics, and performance behavior using these arguments:
- ```backward``` (bool): Controls the temporal direction of the simulation (forward or backward-in-time tracking)
- ```model``` ['CMCC', 'CMEMS', 'POLCOMS']: Specifies the hydrodynamic ocean model
- ```RCP``` (int): Selects the Representative Concentration Pathway greenhouse gas scenario (e.g., 45 or 85) for future climate projection datasets
- ```multi``` (bool): Enables or disables parallel computation capabilities NOT IMPLEMENTED
- ```chunck_n``` (int, default: 1): Defines the number of CPU worker processes or sub-arrays to divide the particle computation NOT IMPLEMENTED
- ``` file``` (str): Defines the path to your simulation options JSON file
- ```config```: Dictionary regulating kinematics and verbosity:
    - ```verbose``` (bool): Toggles terminal execution status messages
    - ```progress_bar``` (bool): Toggles the progress bar visualization during run time
    - ```horizontal_advection``` (bool): Enables/disables horizontal particle tracking via ocean currents
    - ```vertical_advection``` (bool): Enables/disables vertical particle tracking via ocean currents, if available in oceanic dataset
    - ```diffusion``` (bool): (bool): Enables/disables turbulent sub-grid horizontal dispersion
    - ```diffusion coefficient``` (float): Turbulent diffusivity constant ($K$ in $\text{m}^2/\text{s}$)
    - ```diffusion dt``` (int): Time step size used for computing diffusion steps (seconds)
    - ```day depth``` (float): Deepest boundary depth target for migrating larvae during the daytime (meters)
    - ```night depth``` (float): Shallowest boundary depth target for migrating larvae during the daytime (meters)
    - ```dvm speed``` (float): Active vertical migration speed ($\text{m}/\text{s}$)
