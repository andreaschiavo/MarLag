import json_gen as jg
from lagrlib.larvae_generator import LarvaeGenerator


data = {
    "grid" : "data/oceanic.csv",
    "grid margins" : {
        "left":12.,
        "right":23.5,
        "top":45.8,
        "bottom":34.9,
    },  
    "depth margins" : (0,300),
    "field path" :  "data/currents/",# "/data01/mediterraneo/",    "/home/schiavo/storage/CMCC_2010_2020_45/", 
    "check field path" : "/data01/mediterraneo/2020/05/", # "/home/schiavo/storage/CMCC_2010_2020_45/"
    "PLD": 54,
    "PLD var": 5,
    "depth range min": 20,
    "depth range max": 250,
    "spawning grid" : "data/sim_grid20800.csv",
    "total particles" : 100000000,
    "spawning season median" : {"17": [[1, 15], [6, 9]], "18": [[12, 22], [8, 6]]}, # original [[2,5], [6,30]] - [[1,12], [8,27]]
    "spawning season duration" : {"17": [90, 105], "18": [67, 67]},
    "spawning export path": "results/larvae_gen/",
    "spawning points path": "results/larvae_gen/LG_2026-05-29_FW/spawning_points_",
    "lagrangian export path": "results/lagrangian/",
}

json = jg.json_generator(data)
json.savefile("data/sim_options")


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


