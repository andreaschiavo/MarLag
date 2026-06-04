
import numpy as np
from scipy.stats import truncnorm
from global_land_mask import globe
from iteround import saferound

def hori_uni_vert_gauss(particles, spawning_model, velocity_fields):
    active = ~spawning_model.skip
    dens = spawning_model.densities.copy()
    dens[~active] = 0
    if dens.sum() > 0:
        dens = dens / dens.sum()
    raw = dens * particles
    juveniles_per_cell = np.array(saferound(raw, 0), dtype=int)

    x_out = []
    y_out = []
    z_out = []

    for cell in np.where(juveniles_per_cell > 0)[0]:
        juveniles_today = juveniles_per_cell[cell]

        left   = spawning_model.spawning_grid.loc[cell, 'left']
        right  = spawning_model.spawning_grid.loc[cell, 'right']
        bottom = spawning_model.spawning_grid.loc[cell, 'bottom']
        top    = spawning_model.spawning_grid.loc[cell, 'top']

        dmin = max(spawning_model.depth_range[0], spawning_model.spawning_grid.loc[cell,'depth_min'])
        dmax = min(spawning_model.depth_range[1], spawning_model.spawning_grid.loc[cell,'depth_max'])
        if dmin >= dmax:
            dmin, dmax = spawning_model.depth_range[0], spawning_model.depth_range[1]

        mid = (dmin + dmax) / 2
        scale = max(mid - dmin, 1e-6)

        a = (dmin - mid) / scale
        b = (dmax - mid) / scale

        remaining = juveniles_today
        
        counter = 0 # DEBUG
        
        batch_factor = 10

        while remaining > 0:
            batch = remaining * batch_factor

            lon = np.random.uniform(left, right, batch)
            lat = np.random.uniform(bottom, top, batch)
            dep = truncnorm.rvs(a, b, loc=mid, scale=scale, size=batch)

            pts = np.column_stack((lon, lat, dep))

            u = velocity_fields.interp_U(pts)
            v = velocity_fields.interp_V(pts)

            valid = (u != 0) | (v != 0)
            valid &= ~globe.is_land(lat, lon)

            lon_valid = lon[valid]
            lat_valid = lat[valid]
            dep_valid = dep[valid]

            take = min(remaining, lon_valid.size)

            x_out.extend(lon_valid[:take])
            y_out.extend(lat_valid[:take])
            z_out.extend(dep_valid[:take])

            remaining -= take
            
            if counter == 1e4:
                raise RuntimeError("Too many iterations, check parameters or increase batch_factor")
            counter += 1
            
    return x_out, y_out, z_out