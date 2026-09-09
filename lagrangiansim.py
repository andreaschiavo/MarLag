import numpy as np
import pandas as pd
import xarray as xr
from glob import glob
from scipy.interpolate import RegularGridInterpolator
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
from json import load
from multiprocessing import Pool
from global_land_mask.globe import is_land
from scipy.integrate import solve_ivp
from scipy.stats import truncnorm
from time import time
from tqdm import tqdm
import os

interp_U = interp_V = interp_W = None

class LagrangianSimulation():

    def __init__(self, starting_date, duration, file = "data/lagr_options.json", backward = False, model = 'CMEMS', RCP = None, multi = True, chunck_n = 2):
        self.backward = backward
        self.model = model
        self.multi = multi
        self.chunck_n = chunck_n
        self.options = load(open(file))
        
        self.vertical_adv = self.options["lagrangian"].get("vertical advection", False)
        self.vertical_dvm = self.options["lagrangian"].get("dial vertical migration", False)
        self.eval_z = bool(self.vertical_adv or self.vertical_dvm)
        
        self.X = None
        self.Y = None
        self.Z = None

        self.end_date = ()
        self.sim_days = ()
        self.sim_time = ()
        
        self.depth_range = self.options['depth range']
        self.RCP = RCP

        self.xt = ()
        self.yt = ()
        self.zt = ()
        self.z_ref = ()
        self.age = ()
        self.id = ()
        self.doy = ()
        self.spawn_year = ()
        self.particles_in_year = ()
        self.final = np.array([])

        self.U = None
        self.V = None
        self.W = None

        self.calculate_simulationTime(starting_date=starting_date, duration=duration)
        
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
            self.end_date = end_date
            self.sim_days = (S_DATE - end_date).days
            self.sim_time = np.arange(self.sim_days, -1, -1)
        else:
            if SIM_DAYS != 0 and SIM_MONTHS == 0:
                SIM_DAYS -=1
            elif SIM_MONTHS != 0:
                SIM_DAYS -= 1
            end_date = S_DATE+relativedelta(days = SIM_DAYS)
            end_date = end_date+relativedelta(months = SIM_MONTHS)
            end_date = end_date+relativedelta(years = SIM_YEARS)
            self.end_date = end_date
            self.sim_days = (end_date - S_DATE).days
            self.sim_time = np.arange(self.sim_days, -1, -1)
 
    def load_monthlyFields(self, today):
        M2DEG = 1/111319.494  # meters to degrees
        SEC2DAY = 60*60*24 # seconds to days
        left, right, top, bottom = list(self.options['grid margins'].values())
        D_min, D_max = list(self.options['depth margins'])

        folder = self.options['field path']
        
        date = today - timedelta(days=1) if self.backward else today
        year = date.year
        month = date.month
        
        match self.model:
            case "CMEMS":
                file_list = sorted(glob(f"{folder}{year:04d}/{month:02d}/{year:04d}{month:02d}*_d-CMCC--RFVL-MFSe3r1-MED-b20200901_re-sv01.00.nc"))

                ds = xr.open_mfdataset(file_list, combine = "nested", concat_dim="time").sel(lon = slice(left, right), lat = slice(bottom, top), depth = slice(D_min, D_max)).fillna(0)
                ds = ds * (M2DEG * SEC2DAY) * -1 if self.backward else ds * (M2DEG * SEC2DAY)
                
                U = np.array(ds.uo.values, dtype=np.float32)
                V = np.array(ds.vo.values, dtype=np.float32)

                if self.X is None:
                    X = np.array(ds.lon.values, dtype=np.float32)
                    Y = np.array(ds.lat.values, dtype=np.float32)
                    Z = np.array(ds.depth.values, dtype=np.float32)
                    
            case "POLCOMS":
                path_uo = f"{folder}POLCOMS_ERSEM_biogeochemical-daily-all-rcp{self.RCP:02d}-uo-{year:04d}-{month:02d}-v1.1.nc"
                path_vo = f"{folder}POLCOMS_ERSEM_biogeochemical-daily-all-rcp{self.RCP:02d}-vo-{year:04d}-{month:02d}-v1.1.nc"
                
                ds_uo = xr.open_dataset(path_uo).sel(longitude = slice(left, right), latitude = slice(bottom, top), depth = slice(0, D_max)).fillna(0)
                ds_vo = xr.open_dataset(path_vo).sel(longitude = slice(left, right), latitude = slice(bottom, top), depth = slice(0, D_max)).fillna(0)
                
                ds_uo = ds_uo * (M2DEG * SEC2DAY) * -1 if self.backward else ds_uo * (M2DEG * SEC2DAY)
                ds_vo = ds_vo * (M2DEG * SEC2DAY) * -1 if self.backward else ds_vo * (M2DEG * SEC2DAY)
                
                U = np.array(ds_uo.uo.values, dtype=np.float32)
                V = np.array(ds_vo.vo.values, dtype=np.float32)

                if self.X is None:
                    X = np.array(ds.longitude.values, dtype=np.float32)
                    Y = np.array(ds.latitude.values, dtype=np.float32)
                    Z = np.array(ds.depth.values, dtype=np.float32)
            
            case "CMCC":
                path_vozocrtx = sorted(glob(f"{folder}medsea-cmip5-projections-physics-RCP{self.RCP:02d}-{year:04d}-{month:02d}-*/vozocrtx_medsea-cmip5-projections-physics_RCP{self.RCP:2d}_grid_U.nc"))
                path_vomecrty = sorted(glob(f"{folder}medsea-cmip5-projections-physics-RCP{self.RCP:02d}-{year:04d}-{month:02d}-*/vomecrty_medsea-cmip5-projections-physics_RCP{self.RCP:2d}_grid_V.nc"))
                
                ds_vozo = xr.open_mfdataset(path_vozocrtx, combine = "nested", concat_dim="time").sel(lon = slice(left, right), lat = slice(bottom, top), depth = slice(D_min, D_max)).fillna(0)
                ds_vome = xr.open_mfdataset(path_vomecrty, combine = "nested", concat_dim="time").sel(lon = slice(left, right), lat = slice(bottom, top), depth = slice(D_min, D_max)).fillna(0)

                ds_vozo = ds_vozo * (M2DEG * SEC2DAY) * -1 if self.backward else ds_vozo * (M2DEG * SEC2DAY)
                ds_vome = ds_vome * (M2DEG * SEC2DAY) * -1 if self.backward else ds_vome * (M2DEG * SEC2DAY)
                
                U = np.array(ds_vozo.vozocrtx.values, dtype=np.float32)
                V = np.array(ds_vome.vomecrty.values, dtype=np.float32)

                if self.X is None:
                    X = np.array(ds_vozo.lon.values, dtype=np.float32)
                    Y = np.array(ds_vozo.lat.values, dtype=np.float32)
                    Z = np.array(ds_vozo.depth.values, dtype=np.float32)

                if self.vertical_adv:
                    path_vovecrtz = sorted(glob(f"{folder}medsea-cmip5-projections-physics-RCP{self.RCP:02d}-{year:04d}-{month:02d}-*/vovecrtz_medsea-cmip5-projections-physics_RCP{self.RCP:2d}_grid_W.nc"))
                    ds_vove = xr.open_dataset(path_vovecrtz).sel(lon = slice(left, right), lat = slice(bottom, top), depth = slice(D_min, D_max)).fillna(0)
                    
                    ds_vove = ds_vove * (SEC2DAY) if self.backward else ds_vove * (SEC2DAY) *-1
                    
                    W = np.array(ds_vove.vovecrtz.values, dtype=np.float32)
                
        if self.X is None:
            self.X = X
            self.Y = Y
            self.Z = Z
            
        self.U_month = U
        self.V_month = V
        self.W_month = W if (self.vertical_adv and self.model == 'CMCC') else ()
     
    def load_dailyFields(self, today):
        M2DEG = 1/111319.494  # meters to degrees
        SEC2DAY = 60*60*24 # seconds to days
        left, right, top, bottom = list(self.options['grid margins'].values())
        D_min, D_max = list(self.options['depth margins'])

        folder = self.options['field path']
        date = today - timedelta(days=1) if self.backward else today
        year = date.year
        month = date.month
        day = date.day

        if self.model == 'CMEMS':
            path = f"{folder}{year:04d}/{month:02d}/{year:04d}{month:02d}{day:02d}_d-CMCC--RFVL-MFSe3r1-MED-b20200901_re-sv01.00.nc"
            
            ds = xr.open_dataset(path).sel(lon = slice(left, right), lat = slice(bottom, top), depth = slice(D_min, D_max)).fillna(0)
            ds = ds * (M2DEG * SEC2DAY) * -1 if self.backward else ds * (M2DEG * SEC2DAY)
            
            U = np.array(ds.uo.values, dtype=np.float32)[0]
            V = np.array(ds.vo.values, dtype=np.float32)[0]

            if self.X is None:
                X = np.array(ds.lon.values, dtype=np.float32)
                Y = np.array(ds.lat.values, dtype=np.float32)
                Z = np.array(ds.depth.values, dtype=np.float32)


        elif self.model == 'POLCOMS':
            path_uo = f"{folder}POLCOMS_ERSEM_biogeochemical-daily-all-rcp{self.RCP:02d}-uo-{year:04d}-{month:02d}-v1.1.nc"
            path_vo = f"{folder}POLCOMS_ERSEM_biogeochemical-daily-all-rcp{self.RCP:02d}-vo-{year:04d}-{month:02d}-v1.1.nc"
            
            ds_uo = xr.open_dataset(path_uo).sel(longitude = slice(left, right), latitude = slice(bottom, top), depth = slice(0, D_max)).fillna(0)
            ds_vo = xr.open_dataset(path_vo).sel(longitude = slice(left, right), latitude = slice(bottom, top), depth = slice(0, D_max)).fillna(0)
            
            ds_uo = ds_uo * (M2DEG * SEC2DAY) * -1 if self.backward else ds_uo * (M2DEG * SEC2DAY)
            ds_vo = ds_vo * (M2DEG * SEC2DAY) * -1 if self.backward else ds_vo * (M2DEG * SEC2DAY)
            
            U = np.array(ds_uo.uo.values, dtype=np.float32)[day-1]
            V = np.array(ds_vo.vo.values, dtype=np.float32)[day-1]

            if self.X is None:
                X = np.array(ds.longitude.values, dtype=np.float32)
                Y = np.array(ds.latitude.values, dtype=np.float32)
                Z = np.array(ds.depth.values, dtype=np.float32)

        elif self.model == 'CMCC':
            path_vozocrtx = glob(f"{folder}medsea-cmip5-projections-physics-RCP{self.RCP:02d}-{year:04d}-{month:02d}-{day:02d}/vozocrtx_medsea-cmip5-projections-physics_RCP{self.RCP:2d}_grid_U.nc")[0]
            path_vomecrty = glob(f"{folder}medsea-cmip5-projections-physics-RCP{self.RCP:02d}-{year:04d}-{month:02d}-{day:02d}/vomecrty_medsea-cmip5-projections-physics_RCP{self.RCP:2d}_grid_V.nc")[0]

            ds_vozo = xr.open_dataset(path_vozocrtx).sel(lon = slice(left, right), lat = slice(bottom, top), depth = slice(D_min, D_max)).fillna(0)
            ds_vome = xr.open_dataset(path_vomecrty).sel(lon = slice(left, right), lat = slice(bottom, top), depth = slice(D_min, D_max)).fillna(0)

            ds_vozo = ds_vozo * (M2DEG * SEC2DAY) * -1 if self.backward else ds_vozo * (M2DEG * SEC2DAY)
            ds_vome = ds_vome * (M2DEG * SEC2DAY) * -1 if self.backward else ds_vome * (M2DEG * SEC2DAY)
            
            U = np.array(ds_vozo.vozocrtx.values, dtype=np.float32)
            V = np.array(ds_vome.vomecrty.values, dtype=np.float32)

            if self.X is None:
                X = np.array(ds_vozo.lon.values, dtype=np.float32)
                Y = np.array(ds_vozo.lat.values, dtype=np.float32)
                Z = np.array(ds_vozo.depth.values, dtype=np.float32)

            if self.vertical_adv:
                path_vovecrtz = glob(f"{folder}medsea-cmip5-projections-physics-RCP{self.RCP:02d}-{year:04d}-{month:02d}-{day:02d}/vovecrtz_medsea-cmip5-projections-physics_RCP{self.RCP:2d}_grid_W.nc")[0]
                ds_vove = xr.open_dataset(path_vovecrtz).sel(lon = slice(left, right), lat = slice(bottom, top), depth = slice(D_min, D_max)).fillna(0)
                
                ds_vove = ds_vove * (SEC2DAY) if self.backward else ds_vove * (SEC2DAY) *-1
                
                W = np.array(ds_vove.vovecrtz.values, dtype=np.float32)

        if self.X is None:
            self.X = X
            self.Y = Y
            self.Z = Z
        
        self.U = U
        self.V = V
        self.W = W if (self.vertical_adv and self.model == 'CMCC') else None

    def _load_dailyFields_from_month(self, today):
        day = today.day
        self.U = self.U_month[day-1]
        self.V = self.V_month[day-1]
        if (self.vertical_adv and self.model == 'CMCC'):
            self.W = self.W_month[day-1]
            
    def load_particles_year(self, year):
        path = f"{self.options['lagrangian']['spawning']}{year:04d}.csv"
        self.particles_in_year = pd.read_csv(path, delimiter = ',', header=0, index_col=None)
        
    def load_particles_day(self, year, doy):
        data = self.particles_in_year.copy()
        data = np.array(data[data.loc[:,"doy"]==doy])

        def new_age(pld, pld_range, n):
            loc = pld
            scale = pld_range/2
            a, b = ((pld-pld_range)-loc)/scale, ((pld+pld_range)-loc)/scale
            return np.array(truncnorm.rvs(a, b, loc = loc, scale = scale, size = n), dtype=int)

        if self.multi:
            chuncks = np.array_split(data, self.chunck_n)
            if not list(self.xt):
                self.xt = [chuncks[i][:,0] for i in range(len(chuncks))]
                self.yt = [chuncks[i][:,1] for i in range(len(chuncks))]
                self.zt = [chuncks[i][:,2] for i in range(len(chuncks))]
                self.doy = [np.full(ch.shape[0], doy, dtype=int) for ch in chuncks]
                self.id = [chuncks[i][:,4] for i in range(len(chuncks))]
                self.spawn_year = [np.full(ch.shape[0], year, dtype=int) for ch in chuncks]
                self.z_ref = [np.copy(ch[:,2]) for ch in chuncks]
                
                self.age = [new_age(self.options['PLD'], 5, chunck.shape[0]) for chunck in chuncks]
            else:
                for i in range(len(chuncks)):
                    self.xt[i] = np.concatenate((self.xt[i], chuncks[i][:,0]))
                    self.yt[i] = np.concatenate((self.yt[i], chuncks[i][:,1]))
                    self.zt[i] = np.concatenate((self.zt[i], chuncks[i][:,2]))
                    self.doy[i] = np.concatenate((self.doy[i], np.full(chuncks[i].shape[0], doy, dtype=int)))
                    self.id[i] = np.concatenate((self.id[i], chuncks[i][:,4]))
                    self.spawn_year[i] = np.concatenate((self.spawn_year[i], np.full(chuncks[i].shape[0], year, dtype=int)))
                    
                    self.z_ref[i] = np.concatenate((self.z_ref[i], chuncks[i][:,2]))
                    
                    self.age[i] = np.concatenate((self.age[i], new_age(self.options['PLD'], 5, chuncks[i].shape[0])))
        else:
            if not list(self.xt):
                self.xt = data[:,0]
                self.yt = data[:,1]
                self.zt = data[:,2]
                self.doy = data[:,3]
                self.id = data[:,4]
                self.spawn_year = np.full(data.shape[0], year, dtype=int)
                
                self.z_ref = np.copy(data[:,2])
                
                self.age = new_age(self.options['PLD'], 5, data.shape[0])
            else:               
                self.xt = np.concatenate((self.xt, data[:,0]))                
                self.yt = np.concatenate((self.yt, data[:,1]))                
                self.zt = np.concatenate((self.zt, data[:,2]))
                self.doy = np.concatenate((self.doy, data[:,3]))
                self.id = np.concatenate((self.id, data[:,4]))
                self.spawn_year = np.concatenate((self.spawn_year, np.full(data.shape[0], year, dtype=int)))
                
                self.z_ref = np.concatenate((self.z_ref, data[:,2]))
                
                self.age = np.concatenate((self.age, new_age(self.options['PLD'], 5, data.shape[0])))

    def multiEngineArgs(self):
        
        args = [(self.eval_z, self.xt[i].size, self.xt[i], self.yt[i], self.zt[i], self.age[i], self.z_ref[i], self.options["lagrangian"]) for i in range(self.chunck_n)]

        return args
    
    def storeFinal(self, index):
        if self.multi:
            for i in range(self.chunck_n):
                if list(index[i]):
                    x = self.xt[i][index[i]]
                    y = self.yt[i][index[i]]
                    z = self.zt[i][index[i]]
                    age = self.age[i][index[i]]
                    doy = self.doy[i][index[i]]
                    id = self.id[i][index[i]]
                    spawn_year = self.spawn_year[i][index[i]]

                    if not list(self.final):
                        self.final = np.array((x,y,z,age,doy,id, spawn_year))
                    else:
                        self.final = np.concatenate((self.final, (x,y,z,age,doy,id, spawn_year)), axis=1)
                
                    self.xt[i] = np.delete(self.xt[i], index[i])
                    self.yt[i] = np.delete(self.yt[i], index[i])
                    self.zt[i] = np.delete(self.zt[i], index[i])
                    self.age[i] = np.delete(self.age[i], index[i])
                    self.doy[i] = np.delete(self.doy[i], index[i])
                    self.id[i] = np.delete(self.id[i], index[i])
                    self.spawn_year[i] = np.delete(self.spawn_year[i], index[i])
                    self.z_ref[i] = np.delete(self.z_ref[i], index[i])
        else:
            if list(index):
                x = self.xt[index]
                y = self.yt[index]
                z = self.zt[index]
                age = self.age[index]
                doy = self.doy[index]
                id = self.id[index]
                spawn_year = self.spawn_year[index]
                if not list(self.final):
                    self.final = np.array((x,y,z,age,doy,id, spawn_year))
                else:
                    self.final = np.concatenate((self.final, (x,y,z,age,doy,id, spawn_year)), axis=1)
                self.xt = np.delete(self.xt, index)
                self.yt = np.delete(self.yt, index)
                self.zt = np.delete(self.zt, index)
                self.age = np.delete(self.age, index)
                self.doy = np.delete(self.doy, index)
                self.id = np.delete(self.id, index)
                self.spawn_year = np.delete(self.spawn_year, index)
                self.z_ref = np.delete(self.z_ref, index)

    def create_outputFolder(self, output = None):
        self.options["lagrangian"]["output final"] = (
            output if output else self.options["lagrangian"]["output final"]
            )
        export = self.options["lagrangian"]["output final"]

        base_export = export
        suffix = 0

        while True:
            try:
                os.mkdir(export)
                break   # folder created successfully, exit the loop
            except FileExistsError:
                export = f"{base_export}_{suffix:02d}"
                suffix += 1

        print(f"Output path set to: {export}")
        return export

    def run_sim(self, output = None):
        export = self.create_outputFolder(output)
        
        start = time()
        current_month = -1
        current_year = -1
        
        p = Pool(
            processes = self.chunck_n,
            initializer = init_worker,
            )
        
        for d in tqdm(self.sim_time):

            today = self.end_date + relativedelta(days = int(d)) if self.backward else self.end_date - relativedelta(days = int(d))
            doy = today.timetuple().tm_yday # day of the year
            if today.year != current_year:
                self.load_particles_year(today.year)
                current_year = today.year
            if today.month != current_month:
                #self.close_fields_memory()
                self.load_monthlyFields(today)
                current_month = today.month

            self.load_particles_day(year = current_year, doy = doy)
            
            self._load_dailyFields_from_month(today)
            
            p.starmap(
                update_interpolators,
                [(self.U, self.V, self.W, self.X, self.Y, self.Z, self.eval_z)] * self.chunck_n
            )
            
            args = self.multiEngineArgs()

            results = p.map(multiprocessing_engine, args)

            self.xt = [item[0] for item in results]
            self.yt = [item[1] for item in results]
            self.zt = [item[2] for item in results]
            self.age = [item[3] for item in results]
            index = [item[4] for item in results]

            self.storeFinal(index)

            if self.backward:
                if today.month == 1 and today.day == 1:
                    self.save_output(today, export)
                    
            else:
                if today.month == 12 and today.day == 31:
                    self.save_output(today, export)
                    
        p.close()
        p.join()
        
        if self.backward:         
            index = [np.arange(0,self.xt[i].size, 1, int) for i in range(len(self.xt))]
            self.storeFinal(index)
            self.save_output(today, export)
        else:
            index = [np.arange(0,self.xt[i].size, 1, int) for i in range(len(self.xt))]
            self.storeFinal(index)
            self.save_output(today, export)

        print(f"Done! time: {time()-start:.2f}s")

    def save_output(self, today, export):
        
        x, y, z, age, doy, pid, spawn_year  = self.final

        unique_id = spawn_year * 10**9 + doy.astype(int) * 10**6 + pid.astype(int)

        out = np.vstack((x, y, z, age, unique_id)).T

        fmt = ["%.6f", "%.6f", "%.6f", "%d", "%d"]
        np.savetxt(f"{export}/final_{today.year:04d}.csv", out, fmt=fmt, delimiter=",")
        self.final = np.array([])
        
def evaluate_run(x, y, z, age):
    # land mask
    land = is_land(y,x)
    # punti (N, 3) per gli interpolatori: (z, y, x)
    points = np.column_stack((z, y, x))  # shape (N, 3)
    u = interp_U(points)  # shape (N,)
    v = interp_V(points)  # shape (N,)
    
    u0 = np.nan_to_num(u) == 0.0
    v0 = np.nan_to_num(v) == 0.0
    zero_vel = u0 & v0

    dead_age = (age <= 0)
    keep = ~(land | dead_age | zero_vel)
    return keep

def multiprocessing_engine(args):
    global interp_U, interp_V, interp_W

    eval_z, nsites, x, y, z, age, z_ref, options = args
    
    t = (0,1)
    dt = np.linspace(0,1,25)

    IC = np.array([x,y,z], dtype=np.float32).flatten() if eval_z else np.array([x,y], dtype=np.float32).flatten()

    f = Displacement(interp_U, interp_V, nsites, interp_W, z, z_ref, options)

    ode = solve_ivp(f, t_span=t, y0=IC, method='RK45', t_eval = dt)

    if eval_z:
        
        x, y, z = ode.y[:nsites, -1], ode.y[nsites:-nsites, -1], ode.y[-nsites:, -1]
    else:
        x, y = ode.y[:nsites, -1], ode.y[nsites:, -1]

    age -= 1

    canc = evaluate_run(x,y,z,age)

    idx = np.where(canc == False)[0]

    return (x,y,z,age, idx)

def update_interpolators(U, V, W, X, Y, Z, eval_z):
        global interp_U, interp_V, interp_W

        interp_U = RegularGridInterpolator((Z, Y, X), U, method='linear', bounds_error=False, fill_value=0.0)
        interp_V = RegularGridInterpolator((Z, Y, X), V, method='linear', bounds_error=False, fill_value=0.0)

        if eval_z and (W is not None):
            interp_W = RegularGridInterpolator((Z, Y, X), W, method='linear', bounds_error=False, fill_value=0.0)
        else:
            interp_W = None

def init_worker():
    global interp_U, interp_V, interp_W
    interp_U = None
    interp_V = None
    interp_W = None

def diffusion_lonlat(lon_deg, lat_deg, K=10.0, dt=86400.0):
    """
    lon_deg, lat_deg : array di longitudine/latitudine in gradi
    K                : diffusività orizzontale [m^2/s]
    dt               : timestep [s]
    """
    lon_deg = np.asarray(lon_deg)
    lat_deg = np.asarray(lat_deg)

    sigma_m = np.sqrt(2 * K * dt)   # [m]

    # random walk in meters
    dX_m = sigma_m * np.random.normal(0.0, 1.0, size=lon_deg.shape)
    dY_m = sigma_m * np.random.normal(0.0, 1.0, size=lat_deg.shape)

    M2DEG = 1.0 / 111319.494       # m → degree lat
    lat_rad = np.deg2rad(lat_deg)

    # m → degree
    dLat_deg = dY_m * M2DEG
    dLon_deg = dX_m * M2DEG / np.cos(lat_rad)

    lon_new = lon_deg + dLon_deg
    lat_new = lat_deg + dLat_deg

    return lon_new, lat_new


class Displacement():
    def __init__(self, interp_U, interp_V, nsites, interp_W = None, z = None, z_ref = None, options = None):
        self.interp_U = interp_U
        self.interp_V = interp_V
        self.interp_W = interp_W
        self.nsites = nsites
        self.z = z
        
        if options is None:
            options = {}
        
        self.vertical_adv = options.get("vertical advection", False)
        self.vertical_dvm = options.get("dial vertical migration", False)
        self.eval_z = bool(self.vertical_adv or self.vertical_dvm)
        
        if self.vertical_adv and interp_W is None:
            raise ValueError("W grid is not called")
        
        if self.vertical_dvm:
            self.z_surface = options.get("dvm surface depth", 5)
            self.dvm_speed = options.get("dvm speed", 0.25)
            self.z_deep = options.get("dvm deep depth", 200) if z_ref is None else z_ref
        
    def z_target(self, t):
        s = 0.5 * (1.0 - np.cos(2.0*np.pi * t)) # scalar
        return self.z_surface + s * (self.z_deep - self.z_surface)
    
    def dvm(self, t, z):
        if self.dvm_speed <= 0.0:
            return 0.0 * z

        z_tar = self.z_target(t)           # array stessa shape di z
        w = (z_tar - z) * self.dvm_speed   # <-- segno giusto, dinamica stabile
        
        #print("z:", z[:5], "z_tar:", z_tar[:5], "w:", w[:5])
        return w

    def advection(self, points, y, U, V, W = None):
        u = U(points)
        v = V(points)
        if W is not None:
            w = W(points)
            return (u / np.cos(np.radians(y)), v, w)
        else:
            return (u / np.cos(np.radians(y)), v)
        
    def __call__(self, t, y0):

        nsites = self.nsites
        
        interp_U = self.interp_U
        interp_V = self.interp_V
        interp_W = self.interp_W

        if self.eval_z:
            x, y, z = y0[:(nsites)], y0[(nsites):(-nsites)], y0[(-nsites):]
        else:
            x, y, z = y0[:(nsites)], y0[(-nsites):], self.z

        points = np.array([z, y, x], dtype=np.float32).T
        
        # ADVECTION        
        if self.vertical_adv:
            u, v, w_adv = self.advection(points, y, interp_U, interp_V, interp_W)
        else:
            u, v = self.advection(points, y, interp_U, interp_V)
            w_adv = 0.0 * z
        ###
        # DIAL VERTICAL MIGRATION
        w_dvm = self.dvm(t, z) if self.vertical_dvm else 0.0 * z
        ###
        w_tot = w_adv + w_dvm
        
        #print(w_tot)
        
        if self.eval_z:
            return np.concatenate((u, v, w_tot), axis=0)
        else:
            return np.concatenate((u, v), axis=0)
        
        