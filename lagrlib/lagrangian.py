import numpy as np
import json
from tqdm import tqdm
import time
from scipy.integrate import solve_ivp

from lagrlib.utils.time import Calendar
from lagrlib.myio.velocity_fields import FieldsManager
from lagrlib.myio.spawning_reader import SpawningReader
from lagrlib.particles.larvae import LarvaeManager
from lagrlib.myio.writer import Exporter
from lagrlib.physics.movement import HorizontalAdvection, VerticalAdvection, Diffusion, DVM
from lagrlib.physics.engine import MovementEngine
import lagrlib.myio.velocity_fields as vf


class Lagrangian:
    def __init__(self,
                 starting_date,
                 duration,
                 backward = False,
                 model = 'CMCC',
                 multi = True,
                 chunck_n = 1,
                 RCP = None,
                 file='data/lagr_options.json',
                 config = None):
        
        self.verbose = config.get('verbose', False) if config else False
        self.progress_bar = config.get('progress_bar', True) if config else True
        
        ds = json.load(open(file))
        self.options = ds
        
        self.backward = backward
        self.multi = multi
        self.model = model
        self.chunck_n = chunck_n
        
        self.calendar = Calendar(starting_date, duration, backward = backward)
        
        horizontal_adv = config.get("horizontal advection", True) if config else True
        vertical_adv = config.get("vertical advection", False) if config else False
        dvm = config.get("dial vertical migration", False) if config else False
        diffusion = config.get("diffusion", False) if config else False
        diff_coeff = config.get("diffusion coefficient", 10.0) if config else 10.0
        diff_dt = config.get("diffusion dt", 86400) if config else 86400
        day_depth = config.get("day depth", 100) if config else 100
        night_depth = config.get("night depth", 5) if config else 5
        dvm_speed = config.get("dvm speed", 1.0) if config else 1.0
        
        self.velocity_fields = FieldsManager(ds, model = model, RCP = RCP, vertical_adv = vertical_adv)
        
        self.engine = MovementEngine()
        
        if horizontal_adv:
            self.engine.add_component(HorizontalAdvection())
        if vertical_adv:
            self.engine.add_component(VerticalAdvection())
        if diffusion:
            self.engine.add_component(Diffusion(K=diff_coeff, dt=diff_dt))
        if dvm:
            self.engine.add_component(DVM(day_depth = day_depth, night_depth = night_depth, speed = dvm_speed))

        self.storage = LarvaeManager()
        self.loader = SpawningReader(ds['spawning points path'])
        self.writer = Exporter(ds, type_ = 'lagrangian')
        
        vf.init_worker()
    
    def run_sequential(self, export_path = None):
        
        if self.verbose:
            print('STEP 1: Initialise folder')
        
        self.writer.write_newFolder(export_path, self.backward)
        
        if self.verbose:
            start = time.time()
            print('STEP 2: Starting simulation')
        
        for d in tqdm(self.calendar.sim_time, disable=not self.progress_bar, desc="Simulating particles"):
            self.calendar.today = self.calendar.get_today(d, backward = self.backward)
            self.calendar.doy = self.calendar.get_doy()
            
            if self.calendar.is_new_year():
                self.loader.year_df = self.loader.read_spawning_points(self.calendar.today.year)
            if self.calendar.is_new_month():
                self.velocity_fields.load_month(self.calendar.today, backward = self.backward)
            
            t0 = time.perf_counter() # DEBUG bottleneck
            
            self.velocity_fields._load_dailyFields_from_month(self.calendar.today)
            
            t1 = time.perf_counter() # DEBUG bottleneck
            
            vf.update_interpolators(
                self.velocity_fields.U, 
                self.velocity_fields.V, 
                self.velocity_fields.W, 
                self.velocity_fields.X_grid, 
                self.velocity_fields.Y_grid, 
                self.velocity_fields.Z_grid
                )
            
            print(f"Interpolator U: {vf.interp_U((13.,40.,20. ))}, V: {vf.interp_V((13.,40.,20. ))}") # DEBUG
            
            t2 = time.perf_counter() # DEBUG bottleneck
            
            new_particles = self.loader.load_spawning_points(
                year = self.calendar.today.year,
                doy = self.calendar.doy,
                pld = self.options['PLD'],
                pld_var = self.options['PLD var'],
            )
            
            t3 = time.perf_counter() # DEBUG bottleneck
            
            self.storage.store_larvae(*new_particles)
            print(f"xt size:{self.storage.xt.size}") # DEBUG

            t4 = time.perf_counter() # DEBUG bottleneck
            
            y_IC, nsites = self.storage.get_initial_conditions_and_nsites()
            
            t5 = time.perf_counter() # DEBUG bottleneck
                        
            ode = solve_ivp(
                fun = self.engine,
                y0 = y_IC,
                t_span = (0,1),
                t_eval = np.linspace(0,1,25),
                method = 'RK45',
                args = (nsites,)
                )
            
            t6 = time.perf_counter() # DEBUG bottleneck
            
            self.storage.update_positions(ode.y)
            
            t7 = time.perf_counter() # DEBUG bottleneck
            
            cut_index = self.storage.evaluate_particles()
            
            t8 = time.perf_counter() # DEBUG bottleneck

            self.storage.store_final(cut_index)
            
            t9 = time.perf_counter() # DEBUG bottleneck
            
            print(
                f"fields={t1-t0:.3f}s "
                f"interp={t2-t1:.3f}s "
                f"spawn={t3-t2:.3f}s "
                f"store={t4-t3:.3f}s "
                f"init={t5-t4:.3f}s"
                f"ode={t6-t5:.3f}s "
                f"store={t7-t6:.3f}s "
                f"eval={t8-t7:.3f}s "
                f"final={t9-t8:.3f}s"
            )
            
            if self.calendar.is_last_day(self.calendar.today, backward = self.backward):
                self.writer.save_particles(year = self.calendar.today.year, particles = self.storage.final)
        
        self.storage.store_final(range(self.storage.xt.size))
        self.writer.save_particles(year = self.calendar.today.year, particles = self.storage.final)      
        
        if self.verbose:
            print(f"Done! time: {time.time()-start:.2f}s")      