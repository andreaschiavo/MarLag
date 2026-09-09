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

        base = path + new_dir if path is not None else self.export + new_dir

        export = base
        n = 0
        while os.path.exists(export): # non sovrascrivere i risultati di run precedenti
            export = '%s_%02d' % (base, n)
            n += 1

        os.mkdir(export) # generate destination folder
        self.export = export # DOPO aver risolto il suffisso, non prima
    
    def save_particles_multi(self, particles, date):
        if self.type_ == 'spawning':
            stack = np.vstack(particles)
            year = date.year
            
            df = pd.DataFrame(stack, columns=['x', 'y', 'z', 'doy', 'id', 'state'])
            
            df.to_csv(f'{self.export}/spawning_points_{year}.csv', sep = ',', index=False, float_format='%.3f')
    
    def save_particles(self, year, particles = None, append = False):
        if self.type_ == 'spawning':
            df = pd.DataFrame(particles, columns=['x', 'y', 'z', 'doy', 'id', 'state'])
            
            df.to_csv(f'{self.export}/spawning_points_{year}.csv', sep = ',', index=False, float_format='%.3f')
        
        elif self.type_ == 'lagrangian':
            x, y, z, age, doy, pid, spawn_year, state = particles
            
            unique_id = spawn_year * 10**9 + doy.astype(int) * 10**6 + pid.astype(int)

            out = np.vstack((x, y, z, age, unique_id, state)).T

            fmt = ["%.6f", "%.6f", "%.6f", "%d", "%d", "%d"]
            
            path = f"{self.export}/final_{year:04d}.csv"
            mode = "ab" if (append and os.path.exists(path)) else "wb"
            with open(path, mode) as f: # in append vanno le particelle ancora attive a fine run
                np.savetxt(f, out, fmt=fmt, delimiter=",")
    
    
        


