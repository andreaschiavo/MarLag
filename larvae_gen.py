import numpy as np
import pandas as pd
from glob import glob
import xarray as xr
from tqdm import tqdm
from datetime import date
from global_land_mask import globe
import os
import json
from dateutil.relativedelta import relativedelta
from iteround import saferound
from multiprocessing import Pool
from scipy.stats import truncnorm

class LarvaeGenerator(object):
     
    def __init__(self, starting_date, duration, file='data/lagr_options.json', backward = False, workers = 1, case = 'HKE_17_18_19', item = 'n_km2'):
        self.data = json.load(open(file))
        ds = self.data
        
        self.depth_range = (ds['depth range'][0], ds['depth range'][1], (ds['depth range'][0]+ds['depth range'][1])/2)
        self.spawning_grid = pd.DataFrame(pd.read_csv(ds['spawning']['grid'], delimiter=',')).reset_index(drop=True)
        self.index = self.spawning_grid.index.values
        self.sim_days = ()
        self.sim_time = ()
        self.backward = backward
        self.starting_date = starting_date
        self.workers = workers
        self.case = case
        self.end_date, self.sim_day, self.sim_time = self.calculate_simulationTime(starting_date, duration)
        print(f"End date: {self.end_date}")
        
        self.densities = self.spawning_grid[item].values         
        
        self.x_cell = self.spawning_grid['X']
        self.y_cell = self.spawning_grid['Y']
        
        self.check_U = ()
        self.check_V = ()
        
        self.X_grid = ()
        self.Y_grid = ()
        self.Z_grid = ()
        
        self.interp_U = ()
        self.interp_V = ()
        
        self.skip = ()
        
        self.particles = ()
        
        self.pdf_normal = ()
        self.pdf_leap = ()
        
        self.check_Fields()
        self.precalculate_interpolationGrid()
        self.valid_cells()
        self.precalculate_pdf(case = case)
        
        self.export = None
         
    def calculate_simulationTime(self, starting_date, duration):
        S_DATE = date(*starting_date)
        
        SIM_YEARS, SIM_MONTHS, SIM_DAYS = duration
        
        if self.backward:
            if SIM_DAYS != 0 and SIM_MONTHS == 0:
                SIM_DAYS -=1
            elif SIM_MONTHS != 0:
                SIM_DAYS -= 1
            end_date = S_DATE-relativedelta(days = SIM_DAYS)
            end_date = end_date-relativedelta(months = SIM_MONTHS)
            end_date = end_date-relativedelta(years = SIM_YEARS)
            sim_days = (end_date - S_DATE).days
            sim_time = np.arange(sim_days, -1, -1)
            
            return end_date, sim_days, sim_time
        else:
            if SIM_DAYS != 0 and SIM_MONTHS == 0:
                SIM_DAYS -=1
            elif SIM_MONTHS != 0:
                SIM_DAYS -= 1
            end_date = S_DATE+relativedelta(days = SIM_DAYS)
            end_date = end_date+relativedelta(months = SIM_MONTHS)
            end_date = end_date+relativedelta(years = SIM_YEARS)
            sim_days = (end_date - S_DATE).days
            sim_time = np.arange(sim_days, -1, -1)
            
            return end_date, sim_days, sim_time

    def check_Fields(self):
        
        # conversion of velocity metric --> radiants
        M2DEG = 1/111319.494  # meters to degrees
        SEC2DAY = 60*60*24 # seconds to days
        
        left, right, top, bottom = list(self.data['grid margins'].values())
        D_min, D_max = list(self.data['depth margins'])
                
        ds = xr.open_dataset(glob(f"{self.data['check field path']}*.nc")[0])
        check_U = ds['uo']
        check_Ugrid = np.array(check_U[0,:,:,:]*M2DEG*SEC2DAY)
        check_V = ds['vo']
        check_Vgrid = np.array(check_V[0,:,:,:]*M2DEG*SEC2DAY)
        
        lon = ds.lon.values
        lat = ds.lat.values
        depth = ds.depth.values
        ds.close()
        
        lo1 = np.argmin(np.abs(lon-left))
        lo2 = np.argmin(np.abs(lon-right))
        la1 = np.argmin(np.abs(lat-bottom))
        la2 = np.argmin(np.abs(lat-top))
        de1 = np.argmin(np.abs(depth-D_min))
        de2 = np.argmin(np.abs(depth-D_max))
        
        check_Ugrid = np.transpose(check_Ugrid, (2,1,0))[lo1:lo2,la1:la2, de1:de2] #re-arrange the array: long, lat, depth
        check_Vgrid = np.transpose(check_Vgrid, (2,1,0))[lo1:lo2,la1:la2, de1:de2]
        
        check_Ugrid[check_Ugrid>1e2] = 0
        check_Vgrid[check_Vgrid>1e2] = 0
        
        self.check_U = check_Ugrid
        self.check_V = check_Vgrid
        
        self.X_grid = lon[lo1:lo2]
        self.Y_grid = lat[la1:la2]
        self.Z_grid = depth[de1:de2]
    
    def precalculate_interpolationGrid(self):
        from scipy.interpolate import RegularGridInterpolator
        self.interp_U = RegularGridInterpolator(
            (self.X_grid, self.Y_grid, self.Z_grid),
            self.check_U,
            bounds_error=False,
            fill_value=0.0,
        )

        self.interp_V = RegularGridInterpolator(
            (self.X_grid, self.Y_grid, self.Z_grid),
            self.check_V,
            bounds_error=False,
            fill_value=0.0,
        )
    
    def valid_cells(self):
        x = self.x_cell
        y = self.y_cell
        D_range = self.depth_range
        grid = np.array(self.spawning_grid[['depth_min', 'depth_max', 'depth_mean']])
        
        N_cell = self.spawning_grid.shape[0]
        
        skip = []
        
        for cell in range(N_cell):
            upper = [x[cell], y[cell], D_range[0]]
            lower = [x[cell], y[cell]
                     , D_range[1]]
            
            u_up = self.interp_U(upper)
            u_down = self.interp_U(lower)
            
            if grid[cell, 0] > D_range[1] or (u_up == 0 and u_down == 0):
                skip.append(True)
            else:
                skip.append(False)
        self.skip = np.array(skip)
        
        self.densities[self.skip] = 0.0
        self.densities = self.densities / self.densities.sum()

    def check_position(self, x, y, z):
        if (self.interp_U((x,y,z))[0] == 0 and\
            self.interp_V((x,y,z))[0] == 0) or globe.is_land(y,x):
            return False
        else:
            return True
     
    def generate_random_point(self, cell):
        from scipy.stats import truncnorm
        grid = self.spawning_grid
        
        bio_min, bio_max, _ = self.depth_range

        cell_min = grid.loc[cell, 'depth_min']
        cell_max = grid.loc[cell, 'depth_max']

        MIN_DEPTH = max(bio_min, cell_min)
        MAX_DEPTH = min(bio_max, cell_max)

        if MIN_DEPTH >= MAX_DEPTH:
            MIN_DEPTH = bio_min
            MAX_DEPTH = bio_max

        MID_DEPTH = (MIN_DEPTH + MAX_DEPTH) / 2
        
        lonPoint = np.random.uniform(grid.loc[cell,'left'],grid.loc[cell,'right'])
        latPoint = np.random.uniform(grid.loc[cell,'bottom'],grid.loc[cell,'top'])

        scale =  MID_DEPTH-MIN_DEPTH
        a, b = (MIN_DEPTH-MID_DEPTH)/scale, (MAX_DEPTH-MID_DEPTH)/scale
        #depPoint = random.uniform(minDepth, maxDepth)
        depPoint = truncnorm.rvs(a, b, loc = MID_DEPTH, scale = scale)

        return lonPoint, latPoint, depPoint
    
    def store_initialCoord(self, day, x0, y0, z0):
    
        new_date = np.empty(x0.size)
        new_date.fill(int(day))
        id = np.arange(1, (x0.size)+1, 1)

        new_particles0 = np.column_stack((x0, y0, z0, new_date, id ))

        if len(self.particles) == 0:
            self.particles = new_particles0
        else:
            self.particles = np.concatenate((self.particles, new_particles0), axis = 0)

    def precalculate_pdf(self, case = 'HKE_17_18_19'):
        def md_to_doy(md, leap=False):
            # md = [month, day]
            year = 2020 if leap else 2021
            if self.backward:
                doy = (date(year, md[0], md[1]) + relativedelta(days=self.data["PLD"])).timetuple().tm_yday
            else:
                doy = date(year, md[0], md[1]).timetuple().tm_yday
            return doy
        match case:
            case "HKE_17_18_19":
                peak17 = [md_to_doy(self.data['spawning']['season median']['17'][x]) for x in range(2)]
                peak17_leap = [md_to_doy(self.data['spawning']['season median']['17'][x],leap=True) for x in range(2)]
                sigma17 = [self.data['spawning']['season duration']['17'][x] for x in range(2)]
                peak1819 = [md_to_doy(self.data['spawning']['season median']['18'][x]) for x in range(2)]
                peak1819_leap = [md_to_doy(self.data['spawning']['season median']['18'][x], leap=True) for x in range(2)]
                sigma1819 = [self.data['spawning']['season duration']['18'][x] for x in range(2)]
                args = [[peak17, sigma17], [peak1819, sigma1819]]
                args_leap = [[peak17_leap, sigma17], [peak1819_leap, sigma1819]]
                self.pdf_normal = []
                self.pdf_leap = []
                for mu, sigma in args: 
                    self.pdf_normal.append(bimodal_pdf(np.arange(1,366), mu[0], mu[1], sigma[0]/4, sigma[1]/4, period=366))
                for mu, sigma in args_leap:
                    self.pdf_leap.append(bimodal_pdf(np.arange(1,367), mu[0], mu[1], sigma[0]/4, sigma[1]/4, period=367))
                    
    def __call__(self, parent_path=None):

        print('STEP 1: LARVAE GENERATION')

        if self.backward: # generate destination folder path
            new_dir = 'LG_'+str(date.today())+'_BW'
        else:
            new_dir = 'LG_'+str(date.today())+'_FW'

        export = parent_path + new_dir 
        self.export = export # save the new export path

        try:
            os.mkdir(self.export) # generate destination folder
        except FileExistsError:
            if glob(self.export+'*')[-1][-1] == 'W':
                self.export = self.export+'_00'
                os.mkdir(self.export)
            else:
                num = int(glob(self.export+'*')[-1][-2:])
                self.export = self.export+'_%02d' % (num+1,)
                os.mkdir(self.export)

        SPAWNING = self.data['spawning']["particles"] # total number of particles per year

        # build args for multiprocessing
        args = [(
            day,
            self.backward,
            self.end_date,
            self.spawning_grid,
            self.skip,
            self.densities,
            self.depth_range,
            self.interp_U,
            self.interp_V,
            self.pdf_normal,
            self.pdf_leap,
            self.case,
            SPAWNING
            ) for day in self.sim_time]

        particles_all = [] # initialise vector to store particles generated in each cell

        with Pool(self.workers) as p:
            for day, x0, y0, z0 in tqdm(
                p.imap_unordered(process_day, args),
                total=len(args),
                desc="Generating larvae",
                unit="day"
            ):
                if x0.size > 0: # build particle array
                    day_arr = np.full(x0.size, int(day)) # day of spawning from the start of the simulation
                    ids = np.arange(1, x0.size+1) # daily particle ID
                    part = np.column_stack((x0, y0, z0, day_arr, ids)) # assemble particle array
                    particles_all.append(part) # append the new particle data

                # check year boundary
                if self.backward:
                    current_date = self.end_date + relativedelta(days=int(day))
                else:
                    current_date = self.end_date - relativedelta(days=int(day))
                    
                # save the particles generated in the year
                if (self.backward and current_date.month == 1 and current_date.day == 1) or \
                ((not self.backward) and current_date.month == 12 and current_date.day == 31):
                    stack = np.vstack(particles_all)
                    year = current_date.year
                    np.savetxt(f'{self.export}/Initial_positions_{year}.csv',
                                stack, fmt="%.6f", delimiter=",")
                    particles_all = []

            if len(particles_all) > 0: # save remaining particles
                stack = np.vstack(particles_all)
                year = current_date.year
                np.savetxt(f'{self.export}/Initial_positions_{year}.csv',
                            stack, fmt="%.6f", delimiter=",")

        print("Generation complete.")

class ElaborateSpawning(object):
    def __init__(self, path_in, parent_path, end_date, starting_year, backward = False):
        self.path_in = (path_in+'/Initial_positions_')
        if backward:
            self.path_out = (parent_path+'/ES_'+str(date.today()))+'_BW'
        else:
            self.path_out = (parent_path+'/ES_'+str(date.today()))+'_FW'
        self.backward = backward
        
        self.date_ref = end_date
        self.S_YEAR = starting_year
        
        try:
            os.mkdir(self.path_out)
        except FileExistsError:
            if glob(self.path_out+'*')[-1][-1] == 'W':
                self.path_out = self.path_out+'_00'
                os.mkdir(self.path_out)
            else:
                num = int(glob(self.path_out+'*')[-1][-2:])
                self.path_out = self.path_out+'_%02d' % (num+1,)
                os.mkdir(self.path_out)
    
    def compute_dataset(self, ds):

        if self.backward:
            # backward: parto dalla end_date e vado indietro
            for i in tqdm(range(ds.shape[0])):
                ds.iloc[i,3] = (self.date_ref + relativedelta(days = ds.iloc[i,3])).timetuple().tm_yday
        else:
            # forward: parto dalla start_date e vado avanti
            for i in tqdm(range(ds.shape[0])):
                ds.iloc[i,3] = (self.date_ref - relativedelta(days = ds.iloc[i,3])).timetuple().tm_yday

        ds.columns = ['x', 'y', 'z', 'doy', 'id']
        return ds

    def __call__(self):
        print('STEP 2: ELABORATION OF SPAWNING POINTS\n')
        
        if self.backward:
            for i in range((self.S_YEAR - self.date_ref.year)+1):
                dataset_name = (self.path_in+str(self.S_YEAR-i)+'.csv')
                dataset = pd.DataFrame(pd.read_csv(dataset_name, header = None))
                dataset = self.compute_dataset(dataset)
                dataset.to_csv(self.path_out+'/spawning_points_'+str(self.S_YEAR-i)+'.csv', sep = ',', decimal = '.', index = False)
        else:
            for i in range((self.date_ref.year - self.S_YEAR)+1):
                dataset_name = (self.path_in+str(self.S_YEAR+i)+'.csv')
                dataset = pd.DataFrame(pd.read_csv(dataset_name, header = None))
                dataset = self.compute_dataset(dataset)
                dataset.to_csv(self.path_out+'/spawning_points_'+str(self.S_YEAR+i)+'.csv', sep = ',', decimal = '.', index = False)      
            
def process_day(args):
    (
        day,
        backward,
        end_date,
        grid_df,
        skip,
        densities,
        depth_range,
        interp_U,
        interp_V,
        pdf_normal,
        pdf_leap,
        case,
        SPAWNING,
    ) = args
    
    def is_leap(year):
        return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    
    # ---- current date ----
    if backward:
        current_date = end_date + relativedelta(days=int(day))
    else:
        current_date = end_date - relativedelta(days=int(day))

    day_of_year = current_date.timetuple().tm_yday

    match case:
        case "HKE_17_18_19":
            
            valid_grid = grid_df[~skip]
            idx17 = valid_grid[valid_grid['GSA'] == 17].index
            idx1819 = valid_grid[valid_grid['GSA'] > 17].index
            N17_cells = valid_grid.loc[idx17].shape[0]
            N1819_cells = valid_grid.loc[idx1819].shape[0]
            TOT_cells = valid_grid.shape[0]
            
            SPAWNING_17 = SPAWNING * (N17_cells / TOT_cells)
            SPAWNING_1819 = SPAWNING * (N1819_cells / TOT_cells)

            d17   = pdf_leap[0][day_of_year - 1] if is_leap(current_date.year) else pdf_normal[0][day_of_year - 1]
            d1819 = pdf_leap[1][day_of_year - 1] if is_leap(current_date.year) else pdf_normal[1][day_of_year - 1]
            
            larvae_17_today = SPAWNING_17 * d17
            larvae_1819_today = SPAWNING_1819 * d1819
            larvae_today = larvae_17_today + larvae_1819_today

    active = ~skip
    dens = densities.copy()
    dens[~active] = 0
    if dens.sum() > 0:
        dens = dens / dens.sum()
    raw = dens * larvae_today
    juveniles_per_cell = np.array(saferound(raw, 0), dtype=int)

    # ---- generazione particelle ----
    x_out = []
    y_out = []
    z_out = []

    for cell in np.where(juveniles_per_cell > 0)[0]:
        juveniles_today = juveniles_per_cell[cell]

        left   = grid_df.loc[cell, 'left']
        right  = grid_df.loc[cell, 'right']
        bottom = grid_df.loc[cell, 'bottom']
        top    = grid_df.loc[cell, 'top']

        dmin = max(depth_range[0], grid_df.loc[cell,'depth_min'])
        dmax = min(depth_range[1], grid_df.loc[cell,'depth_max'])
        if dmin >= dmax:
            dmin, dmax = depth_range[0], depth_range[1]

        mid = (dmin + dmax) / 2
        scale = max(mid - dmin, 1e-6)

        a = (dmin - mid) / scale
        b = (dmax - mid) / scale

        remaining = juveniles_today
        batch_factor = 10

        while remaining > 0:
            batch = remaining * batch_factor

            lon = np.random.uniform(left, right, batch)
            lat = np.random.uniform(bottom, top, batch)
            dep = truncnorm.rvs(a, b, loc=mid, scale=scale, size=batch)

            pts = np.column_stack((lon, lat, dep))

            u = interp_U(pts)
            v = interp_V(pts)

            valid = (u != 0) | (v != 0)
            valid &= ~globe.is_land(lat, lon)

            lon_valid = lon[valid]
            lat_valid = lat[valid]
            dep_valid = dep[valid]

            take = min(remaining, lon_valid.size)

            x_out.append(lon_valid[:take])
            y_out.append(lat_valid[:take])
            z_out.append(dep_valid[:take])

            remaining -= take

    return (
        day,
        np.concatenate(x_out) if x_out else np.array([]),
        np.concatenate(y_out) if y_out else np.array([]),
        np.concatenate(z_out) if z_out else np.array([]),
    )

def bimodal_pdf(x, mean1, mean2, sigma1, sigma2, period=None, w_base=0.2):
    def gaussian(x, mu, sigma):
        return np.exp(-0.5 * ((x - mu) / sigma)**2)

    # --- bimodal ---
    bim = (
        gaussian(x, mean1, sigma1) +
        gaussian(x, mean2, sigma2)
    )

    if period is not None:
        bim += (
            gaussian(x, mean1 - period, sigma1) +
            gaussian(x, mean1 + period, sigma1) +
            gaussian(x, mean2 - period, sigma2) +
            gaussian(x, mean2 + period, sigma2)
        )

    # --- baseline ---
    base = np.ones_like(x)

    # --- mix ---
    pdf = w_base * base + (1 - w_base) * bim

    # --- normalise ---
    pdf /= pdf.sum()

    return pdf
