import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from datetime import date

from lagrlib.utils.time import is_leap

class SpawningModels:
    def __init__(self, data, item = 'n_km2'):
        
        self.spawning_grid = pd.DataFrame(pd.read_csv(data['spawning grid'], delimiter=',')).reset_index(drop=True)
        self.index = self.spawning_grid.index.values
        
        self.depth_range = (data['depth range min'], data['depth range max'], (data['depth range min']+data['depth range max'])/2)
        self.PLD = data['PLD']
        
        self.x_cell = self.spawning_grid['X']
        self.y_cell = self.spawning_grid['Y']
        
        self.densities = self.spawning_grid[item].values.copy()
        
        self.pdf_normal = ()
        self.pdf_leap = ()
        
        self.skip = ()
        
    def validate_cells(self, velocity_fields):
        self.skip, self.densities = self.valid_cells(velocity_fields)
    
    def valid_cells(self, velocity_fields):
        x = np.array(self.x_cell)
        y = np.array(self.y_cell)
        D_range = self.depth_range
        
        # Estraggo le colonne della griglia: min, max, mean
        depth_min = self.spawning_grid['depth_min'].values
        depth_max = self.spawning_grid['depth_max'].values
        depth_mean = self.spawning_grid['depth_mean'].values
        
        # 1. Controllo geometrico: la cella è troppo profonda per la specie?
        # Se il fondo della cella (depth_min) è sotto il limite massimo della specie (D_range[1])
        cond_quota = depth_min > D_range[1]
        
        # 2. Controllo dinamico (Vettorizzato: niente più ciclo FOR!)
        # Prepariamo i punti tridimensionali usando la profondità MEDIATA della cella
        # che è quella dove le larve passeranno più tempo.
        punti_medi = np.column_stack((x, y, depth_mean))
        
        # Interroghiamo l'interpolatore su TUTTE le celle contemporaneamente
        u_mid = velocity_fields.interp_U(punti_medi)
        v_mid = velocity_fields.interp_V(punti_medi) # Testiamo anche V, non si sa mai!
        
        # Una cella ha moto nullo se sia U che V sono zero alla sua profondità media
        cond_moto_nullo = (u_mid == 0) & (v_mid == 0)
        
        # Se vuoi essere ancora più sicuro, puoi testare anche il limite superiore reale della cella:
        # punti_up = np.column_stack((x, y, np.maximum(depth_min, D_range[0])))
        # u_up = velocity_fields.interp_U(punti_up)
        # ... e unire le condizioni. Ma già la media risolve il 99% dei cicli infiniti.

        # Combiniamo i filtri: saltiamo se la quota è sbagliata OPPURE se il moto è nullo
        skip = cond_quota | cond_moto_nullo
        
        # Aggiorniamo le densità
        densities = self.densities.copy()
        densities[skip] = 0.0
        
        # Evitiamo divisioni per zero se tutte le celle fossero skippate
        somma_dens = densities.sum()
        if somma_dens > 0:
            densities = densities / somma_dens
            
        return skip, densities

class HKE_17_18_19(SpawningModels):
    def __init__(self, data, item = 'n_km2'):
        super().__init__(data, item = item)
    
    def precalculate_pdf(self, data, backward = False):
        
        def md_to_doy(medianday, leap=False):
            # md = [month, day]
            year = 2020 if leap else 2021
            if backward:
                doy = (date(year, medianday[0], medianday[1]) + relativedelta(days=self.PLD)).timetuple().tm_yday
            else:
                doy = date(year, medianday[0], medianday[1]).timetuple().tm_yday
            return doy
        
        peak17 = [
            md_to_doy(
                data['spawning season median']['17'][x]) for x in range(2)
            ]
        peak17_leap = [
            md_to_doy(
                data['spawning season median']['17'][x],leap=True) for x in range(2)
            ]
        sigma17 = [
            data['spawning season duration']['17'][x] for x in range(2)
            ]
        peak1819 = [
            md_to_doy(data['spawning season median']['18'][x]) for x in range(2)
            ]
        peak1819_leap = [
            md_to_doy(data['spawning season median']['18'][x], leap=True) for x in range(2)
            ]
        sigma1819 = [
            data['spawning season duration']['18'][x] for x in range(2)
            ]
        
        args = [[peak17, sigma17], [peak1819, sigma1819]]
        args_leap = [[peak17_leap, sigma17], [peak1819_leap, sigma1819]]
        self.pdf_normal = []
        self.pdf_leap = []
        
        for mu, sigma in args: 
            self.pdf_normal.append(
                bimodal_pdf(
                    np.arange(1,366), mu[0], mu[1], sigma[0]/4, sigma[1]/4, period=366)
                )
        for mu, sigma in args_leap:
            self.pdf_leap.append(
                bimodal_pdf(
                    np.arange(1,367), mu[0], mu[1], sigma[0]/4, sigma[1]/4, period=367)
                )
    
    def get_daily_spawning(self, day, date, total_particles):
        
        valid_grid = self.spawning_grid[~self.skip]
        idx17 = valid_grid[valid_grid['GSA'] == 17].index
        idx1819 = valid_grid[valid_grid['GSA'] > 17].index
        N17_cells = valid_grid.loc[idx17].shape[0]
        N1819_cells = valid_grid.loc[idx1819].shape[0]
        TOT_cells = valid_grid.shape[0]
        
        SPAWNING_17 = total_particles * (N17_cells / TOT_cells)
        SPAWNING_1819 = total_particles * (N1819_cells / TOT_cells)

        d17   = self.pdf_leap[0][day - 1] if is_leap(date.year) else self.pdf_normal[0][day - 1]
        d1819 = self.pdf_leap[1][day - 1] if is_leap(date.year) else self.pdf_normal[1][day - 1]
        
        larvae_17_today = SPAWNING_17 * d17
        larvae_1819_today = SPAWNING_1819 * d1819
        larvae_today = larvae_17_today + larvae_1819_today
        
        return larvae_today

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