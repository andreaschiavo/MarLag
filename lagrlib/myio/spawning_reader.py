import pandas as pd
import numpy as np
from scipy.stats import truncnorm

class SpawningReader:
    def __init__(self, path):
        self.path = path
        self.year_df = None
    
    def read_spawning_points(self, year):
        df = pd.read_csv(f'{self.path}/spawning_points_{year}.csv', delimiter=',', header=0, index_col=None)
        return df
    
    def load_spawning_points(self, year, doy, pld, pld_var, multi = False, chunck_n = 1):
        if self.year_df is None:
            self.year_df = self.read_spawning_points(year)
        data = np.array(self.year_df[self.year_df.loc[:,"doy"]==doy])

        def new_age(pld, pld_range, n):
            loc = pld
            scale = pld_range/2
            a, b = ((pld-pld_range)-loc)/scale, ((pld+pld_range)-loc)/scale
            return np.array(truncnorm.rvs(a, b, loc = loc, scale = scale, size = n), dtype=int)

        xt = data[:,0]
        yt = data[:,1]
        zt = data[:,2]
        doy = data[:,3]
        id = data[:,4]
        spawn_year = np.full(data.shape[0], year, dtype=int)
        age = new_age(pld = pld, pld_range = pld_var, n = data.shape[0]) # generate age for each particle based on a truncated normal distribution
        state = np.zeros(data.shape[0], dtype=int)
        
        return (xt, yt, zt, age, doy, id, spawn_year, state) 