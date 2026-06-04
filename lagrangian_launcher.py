import json_gen as jg
from lagrlib.lagrangian import Lagrangian


data = {
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

json = jg.json_generator(data)
json.savefile("data/sim_options")

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

