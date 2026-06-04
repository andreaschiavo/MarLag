import numpy as np
import pandas as pd
from datetime import date
import os
from glob import glob
from dateutil.relativedelta import relativedelta

class Exporter:
    def __init__(self, data, type_ = 'spawning'):
        self.export = data.get(f'{type_} export path', None)
        self.type_ = type_

    def write_newFolder(self, path, backward = False):
        if backward: # generate destination folder path
            new_dir = 'LG_'+str(date.today())+'_BW' if self.type_ == 'spawning' else 'PT_'+str(date.today())+'_BW'
        else:
            new_dir = 'LG_'+str(date.today())+'_FW' if self.type_ == 'spawning' else 'PT_'+str(date.today())+'_FW'

        export = path + new_dir if path is not None else self.export + new_dir
        self.export = export

        try:
            os.mkdir(export) # generate destination folder
        except FileExistsError:
            if glob(export+'*')[-1][-1] == 'W':
                export = export+'_00'
                os.mkdir(export)
            else:
                num = int(glob(export+'*')[-1][-2:])
                export = export+'_%02d' % (num+1,)
                os.mkdir(export)
    
    def save_particles_multi(self, particles, date):
        if self.type_ == 'spawning':
            stack = np.vstack(particles)
            year = date.year
            
            doy = [date.timetuple().tm_yday] * stack.shape[0] # day of year for all particles
            
            stack[:,3] = doy # replace day of spawning with day of year for easier analysis
            df = pd.DataFrame(stack, columns=['x', 'y', 'z', 'doy', 'id'])
            
            df.to_csv(f'{self.export}/spawning_points_{year}.csv', sep = ',', index=False, float_format='%.3f')
    
    def save_particles(self, year, particles = None):
        if self.type_ == 'spawning':
            df = pd.DataFrame(particles, columns=['x', 'y', 'z', 'doy', 'id'])
            
            df.to_csv(f'{self.export}/spawning_points_{year}.csv', sep = ',', index=False, float_format='%.3f')
        
        elif self.type_ == 'lagrangian':
            x, y, z, age, doy, pid, spawn_year  = particles
            
            unique_id = spawn_year * 10**9 + doy.astype(int) * 10**6 + pid.astype(int)

            out = np.vstack((x, y, z, age, unique_id)).T

            fmt = ["%.6f", "%.6f", "%.6f", "%d", "%d"]
            np.savetxt(f"{self.export}/final_{year:04d}.csv", out, fmt=fmt, delimiter=",")
            self.final = np.array([])
    
    
        


