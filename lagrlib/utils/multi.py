import numpy as np
from multiprocessing import Pool
from scipy.stats import truncnorm
from dateutil.relativedelta import relativedelta
from global_land_mask import globe
from iteround import saferound
from scipy.integrate import solve_ivp

def spawning_args(obj):
    args = [(
            day,
            obj
            ) for day in obj.calendar.sim_time]
    return args

def process_day(args):
    (
        day,
        obj,
    ) = args
    
    # ---- current date ----
    current_date = obj.calendar.get_current_date(day, backward=obj.backward)
    day_of_year = current_date.timetuple().tm_yday

    larvae_today = obj.spawning_model.get_daily_spawning(day_of_year, current_date, obj.data["total particles"])
    
    x_out, y_out, z_out = obj.generator(larvae_today, obj.spawning_model, obj.velocity_fields)
    
    # if x_out is not None and list(x_out):
    #     return (
    #         day,
    #         np.concatenate(x_out) ,
    #         np.concatenate(y_out) ,
    #         np.concatenate(z_out) ,
    #     )
    # else:
    #     return (
    #         day,
    #         np.array([]),
    #         np.array([]),
    #         np.array([]),
    #     )
    if x_out is not None and list(x_out):
        return (
            day,
            np.array(x_out) ,
            np.array(y_out) ,
            np.array(z_out) ,
        )
    else:
        return (
            day,
            np.array([]),
            np.array([]),
            np.array([]),
        )

_CHILD_VF = None
from scipy.interpolate import RegularGridInterpolator

def init_worker(u_data, v_data, w_data, x_grid, y_grid, z_grid, vf_instance):
    """
    Questa funzione viene eseguita UNA SOLA VOLTA da ogni core quando il pool si avvia.
    Costruisce gli interpolatori localmente sul core.
    """
    global _CHILD_VF
    
    # 1. Ricostruiamo gli interpolatori sul processo figlio usando i dati grezzi
    # (Adatta questa parte a come la tua classe 'vf' costruisce gli interpolatori)
    vf_instance.interp_U = RegularGridInterpolator((x_grid, y_grid, z_grid), u_data)
    vf_instance.interp_V = RegularGridInterpolator((x_grid, y_grid, z_grid), v_data)
    vf_instance.interp_W = RegularGridInterpolator((x_grid, y_grid, z_grid), w_data)
    
    # 2. Salviamo l'istanza di vf aggiornata nella variabile globale del figlio
    _CHILD_VF = vf_instance

def tracking_worker(chunk_y0, nsites_chunk, t_span, t_eval, instance):
    
    """
    Questa funzione gira in parallelo su core diversi.
    """
    # solve_ivp vuole una funzione che accetti SOLO (t, y).
    # Usiamo una lambda che prende (t, y) da solve_ivp e chiama il TUO __call__
    # passando il numero specifico di particelle di questo chunk.
    bridge = lambda t, y: instance(t, y, nsites_chunk)
    
    # Lanciamo l'integrazione sul chunk
    sol = solve_ivp(
        fun=bridge,
        t_span=t_span,
        y0=chunk_y0,
        t_eval=t_eval,
        method='RK45'
    )
    
    return sol.y