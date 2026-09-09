import json
from tqdm import tqdm
from multiprocessing import Pool

from lagrlib.utils.time import Calendar
from lagrlib.myio.velocity_fields import FieldsManager
from lagrlib.particles.spawning_models import HKE_17_18_19
from lagrlib.particles.larvae import ParticleManager
from lagrlib.myio.writer import Exporter
from lagrlib.particles.generator import hori_uni_vert_gauss


class LarvaeGenerator(object):
     
    def __init__(self,
                 starting_date,
                 duration,
                 backward = False,
                 model = "CMCC",
                 workers = 1,
                 case = 'HKE_17_18_19',
                 item = 'n_km2',
                 file='data/sim_options.json',
                 config = None,):
        
        self.verbose = config.get('verbose', False) if config else False
        self.progress_bar = config.get('progress_bar', True) if config else True
        
        ds = json.load(open(file))
        self.data = ds
        
        self.backward = backward
        self.workers = workers
        self.case = case

        self.calendar = Calendar(starting_date, duration, backward = backward)
        self.velocity_fields = FieldsManager(ds, model = model)
        self.velocity_fields.load_checkingFields()
        self.velocity_fields.precalculate_interpolationGrid()
        
        match self.case:
            case 'HKE_17_18_19':
                self.spawning_model = HKE_17_18_19(ds, item = item)
                self.generator = hori_uni_vert_gauss
            case _:
                raise ValueError(f"Case {self.case} not recognized.")
        
        self.spawning_model.validate_cells(self.velocity_fields)
        self.spawning_model.precalculate_pdf(ds, self.backward)
        
        self.storage = ParticleManager()
        self.export = Exporter(ds, type_ = 'spawning')
        
        if self.verbose:
            print(f"Spawning model initialised\nEnd date: {self.calendar.end_date}")
    
    def generate_larvae(self, export_path = None):
        
        if self.verbose:
            print('STEP 1: Initialise folder')
        
        self.export.write_newFolder(export_path, self.backward)
        
        
        if self.verbose:
            print('STEP 2: Generate larvae')
        
        if self.workers > 1:
            from lagrlib.utils.multi import spawning_args, process_day
            particles_all = [] # initialise vector to store particles generated in each cell
            
            args = spawning_args(self)
            
            with Pool(self.workers) as p:
                for day, x0, y0, z0 in tqdm(
                    p.imap_unordered(process_day, args),
                    total=len(args),
                    desc="Generating larvae",
                    unit="day",
                    disable=not self.progress_bar,
                ):
                    current_date = self.calendar.get_current_date(day, backward = self.backward)
                    
                    if x0.size > 0:
                        # come nel ramo sequenziale in colonna 3 va il doy, non il contatore
                        parts = self.storage.build_larvae_array(current_date.timetuple().tm_yday, x0, y0, z0)
                        particles_all.append(parts)
                    
                    if self.calendar.is_last_day(current_date, self.backward):
                        if particles_all:
                            self.export.save_particles_multi(particles_all, current_date)
                            particles_all = [] # reset for next batch
                            
            if particles_all: # save any remaining particles
                self.export.save_particles_multi(particles_all, current_date)
        else:
            for day in tqdm(
                self.calendar.sim_time, desc="Generating larvae", unit="day", disable=not self.progress_bar):
                
                current_date = self.calendar.get_current_date(day, backward = self.backward)
                day_of_year = current_date.timetuple().tm_yday

                larvae_today = self.spawning_model.get_daily_spawning(day_of_year, current_date, self.data["total particles"])
                
                x_out, y_out, z_out = self.generator(larvae_today, self.spawning_model, self.velocity_fields)
                
                self.storage.store_larvae(x_out, y_out, z_out, day_of_year)
                
                if self.calendar.is_last_day(current_date, self.backward):
                    if list(self.storage.larvae):
                        self.export.save_particles(current_date.year, self.storage.larvae)
                        self.storage.clear_larvae() # reset for next batch
                            
            if list(self.storage.larvae): # save any remaining particles
                self.export.save_particles(current_date.year, self.storage.larvae)
                self.storage.clear_larvae() # reset for next batch
            
        if self.verbose:
            print("Generation complete.")