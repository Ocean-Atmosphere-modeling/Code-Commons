import os,sys
import xarray as xr
#from roms_tools import TidalForcing
#from roms_tools import Grid

#from roms_tools import RiverForcing, Grid
from roms_tools import Grid, SurfaceForcing
from datetime import datetime,timedelta,date,time

gfile='/home/hunter/roms/romstools/tides/grid_eccofs_3km_08_b7.nc'
outfile="./ERA5_FORCING_ROMSTOOLS_RUTGERS.nc"
#This generates and intermediatge ROMS  grid file for compatibility with ROMS-TOOLS 
prepared_path = "grid_for_romstools.nc"
with xr.open_dataset(gfile) as source:
    prepared = source.load()
    prepared.attrs.pop("title", None)
    prepared.to_netcdf(prepared_path)



grid = Grid(filename=prepared_path)

grid.plot()


start_time=date.today()-timedelta(days=10)
start_time= datetime.combine(start_time, time.min)
end_time=date.today()+timedelta(days=5)
end_time= datetime.combine(end_time, time.min)

surface_forcing_kwargs = {
    "grid": grid,
    "start_time": start_time,
    "end_time": end_time,
    "type": "physics",
    "model_reference_date": datetime(2006, 1, 1), # this is the default
}


surface_forcing = SurfaceForcing(
    **surface_forcing_kwargs,
    source={"name": "ERA5"},
    use_dask=True,
)

surface_forcing.save(outfile)