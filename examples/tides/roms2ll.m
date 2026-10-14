function roms2ll ( input_ncfile, output_ascii_file )
% ROMS2LL:  takes input ROMS netcdf file, writes output lat/lon/time file
%
% USAGE:  roms2ll ( input_ncfile, output_ascii_file )
%
% PARAMETERS:
% Input:
%     input_ncfile:  a ROMS netcdf file
%     output_ascii_file:  latitude and longitude, one point per line
%
%

lon = ncread( input_ncfile, 'lon_rho' );
lat = ncread( input_ncfile, 'lat_rho' );

num_elements = prod(size(lat));
output_data = [lat(:) lon(:)]';

afid = fopen ( output_ascii_file, 'w' );
fprintf ( afid, '%12.8f %12.8f\n', output_data );
fclose ( afid );
