import os,sys
import xarray as xr
from roms_tools import TidalForcing
from roms_tools import Grid
from datetime import datetime

gfile='./GulfOfGuinea_4km.nc'
outfile="GOG_TIDES_TPXO10_ROMSTOOLS_RUTGERS.nc"
%This generates and intermediatge ROMS  grid file for compatibility with ROMS-TOOLS 
prepared_path = "grid_for_romstools.nc"
with xr.open_dataset(gfile) as source:
    prepared = source.load()
    prepared.attrs.pop("title", None)
    prepared.to_netcdf(prepared_path)

grid = Grid(filename=prepared_path)

tpxo_path = "/home/hunter/roms/tides/tpxo10v2_netcdf/DATA/" #Path to the TPXO10 data 
tpxo_dict = {
    "grid": tpxo_path + "grid_tpxo10v2.nc",
    "h": tpxo_path + "h_tpxo10.v2.nc",
    "u": tpxo_path + "u_tpxo10.v2.nc",
}


#Tidal reference date, 
model_reference_date = datetime(2024, 1, 1)
#Create A tidal forcing object using the TPOX data. 
tidal_forcing = TidalForcing(
    grid=grid,
    source={"name": "TPXO", "path": tpxo_dict},
    ntides=15,  # Number of constituents to consider <= 15. Default is 10.
    model_reference_date=model_reference_date,  # Model reference date. Default is January 1, 2000.
    use_dask=True,
    convention="rutgers",
)
# Save the obejcsts data to a NETCF file.  
tidal_forcing.save(outfile)