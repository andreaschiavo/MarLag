from lagrlib.lagrangiansim import LagrangianSimulation
from lagrlib.larvae_gen import LarvaeGenerator, ElaborateSpawning
import json_gen as jg

data = {
    "grid" : "data/oceanic.csv",
    "grid margins" : {
        "left":12.,
        "right":23.5,
        "top":45.8,
        "bottom":34.9,
    },
    "lat": "data/lat.csv",
    "lon": "data/lon.csv",
    "depth": "data/depth.csv",    
    "depth margins" : (0,300),
    "field path" :  "/home/bellu/storage/CMCC_2090_2100_85/",# "/data01/mediterraneo/",    "/home/schiavo/storage/CMCC_2010_2020_45/", 
    "check field path" : "/data01/mediterraneo/2020/05/", # "/home/schiavo/storage/CMCC_2010_2020_45/"
    "PLD": 54,
    "depth range": (20, 250),
    "spawning" : {
        "grid" : "data/sim_grid20800_dvm.csv",
        "particles" : 1000000,
        "season median" : {"17": [[1, 15], [6, 9]], "18": [[12, 22], [8, 6]]}, # original [[2,5], [6,30]] - [[1,12], [8,27]]
        "season duration" : {"17": [90, 105], "18": [67, 67]},
        "output": "results/larvae_gen/",
    },  
    "lagrangian": {
        "spawning": "results/larvae_gen/ES_FF_dvm/spawning_points_",
        "output initial":(),
        "output final":"results/lagrangian/prova",
        "output image":(),
        "vertical advection": False,
        "dial vertical migration": True,
        "dvm surface depth": 5,
        "dvm speed": 10.,
        "dvm deep depth":200,
    },
}
json = jg.json_generator(data)
json.savefile("data/lagr_options")

spawn = LarvaeGenerator(
    starting_date = (2090, 1, 1),
    duration = (10, 4, 0),
    backward = False,
    workers=20,
    item = "spwn",
    case = "HKE_17_18_19",
    file='data/lagr_options.json',
    )

parent_path = 'results/larvae_gen/'
spawn(parent_path)

ES = ElaborateSpawning(
    path_in = spawn.export,
    parent_path = parent_path,
    end_date = spawn.end_date,
    starting_year = 2090,
    backward=spawn.backward,
)
ES()

data["lagrangian"]["spawning"] = f"{ES.path_out}/spawning_points_"
json = jg.json_generator(data)
json.savefile("data/lagr_options")

PLDs = [(54 - round(54*0.5,0)), (54 - round(54*0.2,0)), (54 - round(54*0.1,0)), 54., (54 + round(54*0.1,0))]
#PLDs = [54]

for pld in PLDs:
    data["PLD"] = pld
    json = jg.json_generator(data)
    json.savefile("data/lagr_options")
    lagr = LagrangianSimulation(
        starting_date=(2090,1,1),
        duration = (10,2,0),
        multi = True,
        backward= False,
        chunck_n=8,
        model = "CMCC",
        RCP = 85,
        file = "data/lagr_options.json",
    )
    lagr.run_sim(f"results/lagrangian/forward_FF85_shift-21_PLD{int(pld)}_dvm")