import numpy as np
import xarray as xr
from datetime import timedelta
from glob import glob
from scipy.interpolate import RegularGridInterpolator

interp_U = interp_V = interp_W = None

class FieldsManager(object):
    def __init__(self, options, model = "CMCC", RCP = 45, vertical_adv = False): 
        self.model = model
        self.RCP = RCP if RCP is not None else 45
        self.vertical_adv = vertical_adv
        
        margins = options["grid margins"]
        self.left = margins["left"]
        self.right = margins["right"]
        self.top = margins["top"]
        self.bottom = margins["bottom"]
        self.D_min, self.D_max = options["depth margins"]
        self.path = options["field path"]
        
        self.X_grid = None
        self.Y_grid = None
        self.Z_grid = None
        
        self.U_month = None
        self.V_month = None
        self.W_month = None
        
        self.U = None
        self.V = None
        self.W = None
        
        self.interp_U = None
        self.interp_V = None
        self.interp_W = None
    
    def load_month(self, today, backward = False):
        U, V, W, X, Y, Z = self.read_month(today, backward)
        if self.X_grid is None:
            self.X_grid = X
            self.Y_grid = Y
            self.Z_grid = Z
            
        self.U_month = U
        self.V_month = V
        self.W_month = W
    
    def read_month(self, today, backward = False):
        M2DEG = 1/111319.494  # meters to degrees
        SEC2DAY = 60*60*24 # seconds to days
        scale = (M2DEG * SEC2DAY) * -1 if backward else (M2DEG * SEC2DAY)
        left, right, top, bottom = self.left, self.right, self.top, self.bottom
        D_min, D_max = self.D_min, self.D_max

        U, V, W, X, Y, Z = None, None, None, None, None, None
        
        path = self.path
        
        date = today - timedelta(days=1) if backward else today
        year = date.year
        month = date.month
        
        match self.model:
            case "CMEMS":
                file_list = sorted(glob(f"{path}{year:04d}/{month:02d}/{year:04d}{month:02d}*_d-CMCC--RFVL-MFSe3r1-MED-b20200901_re-sv01.00.nc"))

                ds = xr.open_mfdataset(file_list, combine = "nested", concat_dim="time").sel(lon = slice(left, right), lat = slice(bottom, top), depth = slice(D_min, D_max)).fillna(0)
                ds = ds * scale
                
                U = np.array(ds.uo.values, dtype=np.float32).transpose(0,3,2,1)
                V = np.array(ds.vo.values, dtype=np.float32).transpose(0,3,2,1)

                if self.X_grid is None:
                    X = np.array(ds.lon.values, dtype=np.float32)
                    Y = np.array(ds.lat.values, dtype=np.float32)
                    Z = np.array(ds.depth.values, dtype=np.float32)
                    
            case "POLCOMS":
                path_uo = f"{path}POLCOMS_ERSEM_biogeochemical-daily-all-rcp{self.RCP:02d}-uo-{year:04d}-{month:02d}-v1.1.nc"
                path_vo = f"{path}POLCOMS_ERSEM_biogeochemical-daily-all-rcp{self.RCP:02d}-vo-{year:04d}-{month:02d}-v1.1.nc"
                
                ds_uo = xr.open_dataset(path_uo).sel(longitude = slice(left, right), latitude = slice(bottom, top), depth = slice(0, D_max)).fillna(0)
                ds_vo = xr.open_dataset(path_vo).sel(longitude = slice(left, right), latitude = slice(bottom, top), depth = slice(0, D_max)).fillna(0)
                
                ds_uo = ds_uo * scale
                ds_vo = ds_vo * scale
                
                U = np.array(ds_uo.uo.values, dtype=np.float32).transpose(0,3,2,1)
                V = np.array(ds_vo.vo.values, dtype=np.float32).transpose(0,3,2,1)

                if self.X_grid is None:
                    X = np.array(ds.longitude.values, dtype=np.float32)
                    Y = np.array(ds.latitude.values, dtype=np.float32)
                    Z = np.array(ds.depth.values, dtype=np.float32)
            
            case "CMCC":
                path_vozocrtx = sorted(glob(f"{path}medsea-cmip5-projections-physics-RCP{self.RCP:02d}-{year:04d}-{month:02d}-*/vozocrtx_medsea-cmip5-projections-physics_RCP{self.RCP:2d}_grid_U.nc"))
                path_vomecrty = sorted(glob(f"{path}medsea-cmip5-projections-physics-RCP{self.RCP:02d}-{year:04d}-{month:02d}-*/vomecrty_medsea-cmip5-projections-physics_RCP{self.RCP:2d}_grid_V.nc"))
                
                ds_vozo = xr.open_mfdataset(path_vozocrtx, combine = "nested", concat_dim="time").sel(lon = slice(left, right), lat = slice(bottom, top), depth = slice(D_min, D_max)).fillna(0)
                ds_vome = xr.open_mfdataset(path_vomecrty, combine = "nested", concat_dim="time").sel(lon = slice(left, right), lat = slice(bottom, top), depth = slice(D_min, D_max)).fillna(0)

                ds_vozo = ds_vozo * scale
                ds_vome = ds_vome * scale
                
                U = np.array(ds_vozo.vozocrtx.values, dtype=np.float32).transpose(0,3,2,1)
                V = np.array(ds_vome.vomecrty.values, dtype=np.float32).transpose(0,3,2,1)

                if self.X_grid is None:
                    X = np.array(ds_vozo.lon.values, dtype=np.float32)
                    Y = np.array(ds_vozo.lat.values, dtype=np.float32)
                    Z = np.array(ds_vozo.depth.values, dtype=np.float32)

                if self.vertical_adv:
                    path_vovecrtz = sorted(glob(f"{path}medsea-cmip5-projections-physics-RCP{self.RCP:02d}-{year:04d}-{month:02d}-*/vovecrtz_medsea-cmip5-projections-physics_RCP{self.RCP:2d}_grid_W.nc"))
                    ds_vove = xr.open_dataset(path_vovecrtz).sel(lon = slice(left, right), lat = slice(bottom, top), depth = slice(D_min, D_max)).fillna(0)
                    
                    ds_vove = ds_vove * (SEC2DAY) if backward else ds_vove * (SEC2DAY) *-1
                    
                    W = np.array(ds_vove.vovecrtz.values, dtype=np.float32).transpose(0,3,2,1)
        return U, V, W, X, Y, Z
    
    def load_day(self, today, backward = False):
        U, V, W, X, Y, Z = self.read_day(today, backward)
        if self.X_grid is None:
            self.X_grid = X
            self.Y_grid = Y
            self.Z_grid = Z
            
        self.U = U
        self.V = V
        if (self.vertical_adv and self.model == 'CMCC'):
            self.W = W
    
    def read_day(self, today, backward = False):
        M2DEG = 1/111319.494  # meters to degrees
        SEC2DAY = 60*60*24 # seconds to days
        left, right, top, bottom = self.left, self.right, self.top, self.bottom
        D_min, D_max = self.D_min, self.D_max

        path = self.path
        date = today - timedelta(days=1) if backward else today
        year = date.year
        month = date.month
        day = date.day
        
        U, V, W, X, Y, Z = None, None, None, None, None, None

        if self.model == 'CMEMS':
            path = f"{path}{year:04d}/{month:02d}/{year:04d}{month:02d}{day:02d}_d-CMCC--RFVL-MFSe3r1-MED-b20200901_re-sv01.00.nc"
            
            ds = xr.open_dataset(path).sel(lon = slice(left, right), lat = slice(bottom, top), depth = slice(D_min, D_max)).fillna(0)
            ds = ds * (M2DEG * SEC2DAY) * -1 if backward else ds * (M2DEG * SEC2DAY)
            
            U = np.array(ds.uo.values, dtype=np.float32)[0].transpose(2,1,0)
            V = np.array(ds.vo.values, dtype=np.float32)[0].transpose(2,1,0)

            if self.X_grid is None:
                X = np.array(ds.lon.values, dtype=np.float32)
                Y = np.array(ds.lat.values, dtype=np.float32)
                Z = np.array(ds.depth.values, dtype=np.float32)


        elif self.model == 'POLCOMS':
            path_uo = f"{path}POLCOMS_ERSEM_biogeochemical-daily-all-rcp{self.RCP:02d}-uo-{year:04d}-{month:02d}-v1.1.nc"
            path_vo = f"{path}POLCOMS_ERSEM_biogeochemical-daily-all-rcp{self.RCP:02d}-vo-{year:04d}-{month:02d}-v1.1.nc"
            
            ds_uo = xr.open_dataset(path_uo).sel(longitude = slice(left, right), latitude = slice(bottom, top), depth = slice(0, D_max)).fillna(0)
            ds_vo = xr.open_dataset(path_vo).sel(longitude = slice(left, right), latitude = slice(bottom, top), depth = slice(0, D_max)).fillna(0)
            
            ds_uo = ds_uo * (M2DEG * SEC2DAY) * -1 if backward else ds_uo * (M2DEG * SEC2DAY)
            ds_vo = ds_vo * (M2DEG * SEC2DAY) * -1 if backward else ds_vo * (M2DEG * SEC2DAY)
            
            U = np.array(ds_uo.uo.values, dtype=np.float32)[day-1].transpose(2,1,0)
            V = np.array(ds_vo.vo.values, dtype=np.float32)[day-1].transpose(2,1,0)

            if self.X_grid is None:
                X = np.array(ds.longitude.values, dtype=np.float32)
                Y = np.array(ds.latitude.values, dtype=np.float32)
                Z = np.array(ds.depth.values, dtype=np.float32)

        elif self.model == 'CMCC':
            path_vozocrtx = glob(f"{path}medsea-cmip5-projections-physics-RCP{self.RCP:02d}-{year:04d}-{month:02d}-{day:02d}/vozocrtx_medsea-cmip5-projections-physics_RCP{self.RCP:2d}_grid_U.nc")[0]
            path_vomecrty = glob(f"{path}medsea-cmip5-projections-physics-RCP{self.RCP:02d}-{year:04d}-{month:02d}-{day:02d}/vomecrty_medsea-cmip5-projections-physics_RCP{self.RCP:2d}_grid_V.nc")[0]

            ds_vozo = xr.open_dataset(path_vozocrtx).sel(lon = slice(left, right), lat = slice(bottom, top), depth = slice(D_min, D_max)).fillna(0)
            ds_vome = xr.open_dataset(path_vomecrty).sel(lon = slice(left, right), lat = slice(bottom, top), depth = slice(D_min, D_max)).fillna(0)

            ds_vozo = ds_vozo * (M2DEG * SEC2DAY) * -1 if backward else ds_vozo * (M2DEG * SEC2DAY)
            ds_vome = ds_vome * (M2DEG * SEC2DAY) * -1 if backward else ds_vome * (M2DEG * SEC2DAY)
            
            U = np.array(ds_vozo.vozocrtx.values, dtype=np.float32).transpose(2,1,0)
            V = np.array(ds_vome.vomecrty.values, dtype=np.float32).transpose(2,1,0)

            if self.X_grid is None:
                X = np.array(ds_vozo.lon.values, dtype=np.float32)
                Y = np.array(ds_vozo.lat.values, dtype=np.float32)
                Z = np.array(ds_vozo.depth.values, dtype=np.float32)

            if self.vertical_adv:
                path_vovecrtz = glob(f"{path}medsea-cmip5-projections-physics-RCP{self.RCP:02d}-{year:04d}-{month:02d}-{day:02d}/vovecrtz_medsea-cmip5-projections-physics_RCP{self.RCP:2d}_grid_W.nc")[0]
                ds_vove = xr.open_dataset(path_vovecrtz).sel(lon = slice(left, right), lat = slice(bottom, top), depth = slice(D_min, D_max)).fillna(0)
                
                ds_vove = ds_vove * (SEC2DAY) if backward else ds_vove * (SEC2DAY) *-1
                
                W = np.array(ds_vove.vovecrtz.values, dtype=np.float32).transpose(2,1,0)
        return U, V, W, X, Y, Z
    
    def load_checkingFields(self,):
        U, V, W, X, Y, Z = self.read_checkingFields()
        if self.X_grid is None:
            self.X_grid = X
            self.Y_grid = Y
            self.Z_grid = Z
            
        self.U = U
        self.V = V
           
    def read_checkingFields(self,):
        M2DEG = 1/111319.494  # meters to degrees
        SEC2DAY = 60*60*24 # seconds to days
        left, right, top, bottom = self.left, self.right, self.top, self.bottom
        D_min, D_max = self.D_min, self.D_max

        path = self.path
        
        U, V, W, X, Y, Z = None, None, None, None, None, None
        
        match self.model:
            case "CMEMS":
                file = sorted(glob(f"{path}*/*/*.nc"))[0]

                ds = xr.open_dataset(file).sel(lon = slice(left, right), lat = slice(bottom, top), depth = slice(D_min, D_max)).fillna(0)
                ds = ds * (M2DEG * SEC2DAY)
                
                U = np.array(ds.uo.values, dtype=np.float32).transpose(2,1,0)
                V = np.array(ds.vo.values, dtype=np.float32).transpose(2,1,0)

                if self.X_grid is None:
                    X = np.array(ds.lon.values, dtype=np.float32)
                    Y = np.array(ds.lat.values, dtype=np.float32)
                    Z = np.array(ds.depth.values, dtype=np.float32)
        
            case "POLCOMS":
                file = sorted(glob(f"{path}*.nc"))[0]

                ds = xr.open_dataset(file).sel(lon = slice(left, right), lat = slice(bottom, top), depth = slice(D_min, D_max)).fillna(0)
                ds = ds * (M2DEG * SEC2DAY)
                
                U = np.array(ds.uo.values, dtype=np.float32)[0].transpose(2,1,0)
                V = np.array(ds.vo.values, dtype=np.float32)[0].transpose(2,1,0)

                if self.X_grid is None:
                    X = np.array(ds.longitude.values, dtype=np.float32)
                    Y = np.array(ds.latitude.values, dtype=np.float32)
                    Z = np.array(ds.depth.values, dtype=np.float32)
            
            case "CMCC":
                file_vozocrtx = sorted(glob(f"{path}*/vozocrtx_medsea-cmip5-projections-physics_RCP*_grid_U.nc"))[0]
                file_vomecrty = sorted(glob(f"{path}*/vomecrty_medsea-cmip5-projections-physics_RCP*_grid_V.nc"))[0]
                
                ds_vozo = xr.open_dataset(file_vozocrtx).sel(lon = slice(left, right), lat = slice(bottom, top), depth = slice(D_min, D_max)).fillna(0)
                ds_vome = xr.open_dataset(file_vomecrty).sel(lon = slice(left, right), lat = slice(bottom, top), depth = slice(D_min, D_max)).fillna(0)
                ds_vozo = ds_vozo * (M2DEG * SEC2DAY)
                ds_vome = ds_vome * (M2DEG * SEC2DAY)
                
                U = np.array(ds_vozo.vozocrtx.values, dtype=np.float32).transpose(2,1,0)
                V = np.array(ds_vome.vomecrty.values, dtype=np.float32).transpose(2,1,0)
                
                if self.X_grid is None:
                    X = np.array(ds_vozo.lon.values, dtype=np.float32)
                    Y = np.array(ds_vozo.lat.values, dtype=np.float32)
                    Z = np.array(ds_vozo.depth.values, dtype=np.float32)
            
        return U, V, W, X, Y, Z
        
    def _load_dailyFields_from_month(self, today):
        day = today.day
        self.U = self.U_month[day-1]
        self.V = self.V_month[day-1]
        if (self.vertical_adv and self.model == 'CMCC'):
            self.W = self.W_month[day-1]
    
    def precalculate_interpolationGrid(self):
        
        self.interp_U = RegularGridInterpolator(
            (self.X_grid, self.Y_grid, self.Z_grid),
            self.U,
            bounds_error=False,
            fill_value=0.0,
        )

        self.interp_V = RegularGridInterpolator(
            (self.X_grid, self.Y_grid, self.Z_grid),
            self.V,
            bounds_error=False,
            fill_value=0.0,
        )
        
        if (self.vertical_adv and self.model == 'CMCC'):
            self.interp_W = RegularGridInterpolator(
                (self.X_grid, self.Y_grid, self.Z_grid),
                self.W,
                bounds_error=False,
                fill_value=0.0,
            )
            

def update_interpolators(U, V, W, X, Y, Z):
    global interp_U, interp_V, interp_W

    interp_U = RegularGridInterpolator((X, Y, Z), U, method='linear', bounds_error=False, fill_value=0.0)
    interp_V = RegularGridInterpolator((X, Y, Z), V, method='linear', bounds_error=False, fill_value=0.0)

    if W is not None:
        interp_W = RegularGridInterpolator((X, Y, Z), W, method='linear', bounds_error=False, fill_value=0.0)
    else:
        interp_W = None

def init_worker():
    global interp_U, interp_V, interp_W
    interp_U = None
    interp_V = None
    interp_W = None