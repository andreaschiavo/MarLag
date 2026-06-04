import numpy as np
from global_land_mask import globe

import lagrlib.myio.velocity_fields as vf

from dataclasses import dataclass

@dataclass
class ParticleState:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    age: np.ndarray


class ParticleManager:
    def __init__(self):       
        self.larvae = np.array([])
    
    def store_initialCoord(self, day, x0, y0, z0):
    
        new_date = np.empty(x0.size)
        new_date.fill(int(day))
        id = np.arange(1, (x0.size)+1, 1)

        new_particles0 = np.column_stack((x0, y0, z0, new_date, id ))

        if len(self.larvae) == 0:
            self.larvae = new_particles0
        else:
            self.larvae = np.concatenate((self.larvae, new_particles0), axis = 0)
    
    def check_position(self, x, y, z, velocity_fields):
        if (velocity_fields.interp_U((x,y,z))[0] == 0 and\
            velocity_fields.interp_V((x,y,z))[0] == 0) or globe.is_land(y,x):
            return False
        else:
            return True
        
    def build_larvae_array(self, day, x0, y0, z0):
        day_arr = np.full(x0.size, int(day), dtype=np.int32) # day of spawning from the start of the simulation
        ids = np.arange(1, x0.size+1, dtype=np.int32) # daily particle ID
        part = np.column_stack((x0, y0, z0, day_arr, ids))
        return part
    
    def store_larvae(self, x, y, z, day):
        
        larvae_array = self.build_larvae_array(day, np.array(x, dtype=np.float32), np.array(y, dtype=np.float32), np.array(z, dtype=np.float32))
        
        if not list(self.larvae):
            self.larvae = larvae_array
        else:
            self.larvae = np.concatenate((self.larvae, larvae_array), axis=0)
    
    def clear_larvae(self):
        self.larvae = np.array([])
        
class LarvaeManager(ParticleManager):
    
    def __init__(self):
        super().__init__()
        self.xt = ()
        self.yt = ()
        self.zt = ()
        self.age = ()
        self.id = ()
        self.doy = ()
        self.spawn_year = ()
        
        self.final = np.array([])
                
    def get_initial_conditions_and_nsites(self): # takes stored particles positions and format them for ode
        # step 1: concatenate to get np.array of shape (3, n_particles). 3 is for x, y, z
        ic_matrix = np.concatenate((self.xt[None,:], self.yt[None, :], self.zt[None, :]), axis = 0)
        return (ic_matrix.flatten(), ic_matrix.shape[1])
    
    def update_positions(self, ode_solution):
        self.xt = ode_solution[:,-1][:self.xt.size]
        self.yt = ode_solution[:,-1][self.xt.size:2*self.xt.size]
        self.zt = ode_solution[:,-1][2*self.xt.size:]
    
    def evaluate_particles(self):
        points = np.column_stack((
            self.xt, self.yt, self.zt
        ))
        u = vf.interp_U(points)
        v = vf.interp_V(points)
        
        index = np.where((u == 0) & (v == 0) | globe.is_land(self.yt, self.xt))[0]
        
        return index
            
    def store_larvae(self, xt, yt, zt, age, doy, id, spawn_year):
        
        if list(self.xt):
            self.xt = np.concatenate((self.xt, xt))
            self.yt = np.concatenate((self.yt, yt))
            self.zt = np.concatenate((self.zt, zt))
            self.doy = np.concatenate((self.doy, doy))
            self.id = np.concatenate((self.id, id))
            self.spawn_year = np.concatenate((self.spawn_year, spawn_year))
            self.age = np.concatenate((self.age, age))
        else:
            self.xt = xt
            self.yt = yt
            self.zt = zt
            self.doy = doy
            self.id = id
            self.spawn_year = spawn_year
            self.age = age
            
    def store_final(self, cut_index):
        if list(cut_index):
            x = self.xt[cut_index]
            y = self.yt[cut_index]
            z = self.zt[cut_index]
            age = self.age[cut_index]
            doy = self.doy[cut_index]
            id = self.id[cut_index]
            spawn_year = self.spawn_year[cut_index]
            if not list(self.final):
                self.final = np.array((x,y,z,age,doy,id, spawn_year))
            else:
                self.final = np.concatenate((self.final, (x,y,z,age,doy,id, spawn_year)), axis=1)
            self.xt = np.delete(self.xt, cut_index)
            self.yt = np.delete(self.yt, cut_index)
            self.zt = np.delete(self.zt, cut_index)
            self.age = np.delete(self.age, cut_index)
            self.doy = np.delete(self.doy, cut_index)
            self.id = np.delete(self.id, cut_index)
            self.spawn_year = np.delete(self.spawn_year, cut_index)
    
    def get_saving_features(self):
        return self.xt, self.yt, self.zt, self.age, self.doy, self.id, self.spawn_year