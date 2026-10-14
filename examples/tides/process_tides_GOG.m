% gfile='/home/om/roms/barnegat/grid_bbay_v7m.nc';
% ofile='tides_tpxo_bbay_v7m.nc';

% gfile='/home/om/roms/eccofs/grid_eccofs_3km_01.nc';
% ofile='tides_tpxo_eccofs_3km_01.nc';
% harmonics_prefix='otps_harmonics_eccofs_3km_';
% llfile='./ll_eccofs_3km.dat';
% %
% gfile='/home/om/roms/eccofs/grid_eccofs_6km_01.nc';
% ofile='tides_tpxo_eccofs_6km_01.nc';
% harmonics_prefix='otps_harmonics_eccofs_6km_';
% llfile='./ll_eccofs_6km.dat';
%%
gfile='/home/om/roms/coessing/inp/GulfOfGuinea_4km.nc';
ofile='tides_tpxo_GulfOfGuinea_4km.nc';
harmonics_prefix='otps_harmonics_GulfOfGuinea_4km_';
llfile='./ll_GulfOfGuinea_4km.dat';



addpath /home/om/cron/tides/OTPS/work/TMD_Matlab_Toolbox_v2.5/TMD
addpath /home/om/cron/tides/OTPS/work/TMD_Matlab_Toolbox_v2.5/TMD/FUNCTIONS
addpath /home/om/cron/tides/OTPS/work/t_tide_v1.3beta
addpath /home/hunter/matlab/roms_wilkin/matlab
addpath /home/hunter/matlab/roms1125/matlab/utility
addpath /home/hunter/matlab/mexcdf/snctools
addpath /home/hunter/matlab/mexcdf/mexnc
addpath /home/hunter/matlab/map
addpath /home/hunter/matlab/map/North_Atlantic

base_date=datenum(2024,1,1);
pred_date=datenum(2024,1,1);




model_file='/home/om/cron/tides/OTPS/work/DATA/Model_tpxo9_atlas_v5'

vars=['z','u','v'];
%Build grid file

roms2ll(gfile,llfile);
disp(['Gridfile ' gfile ' lat/lon written to ' llfile])

if ~(exist(model_file,'file'))
    error(['No Such model file: ' model_file])
end

disp(['Mode parameter file used: ' model_file])

run_EX=false;
if run_EX
    for i=1:length(vars)
        disp(['Generating parameter file for ' vars(i)])
        fid=fopen('otps_input','w');
        fprintf(fid,'%s\n',model_file);
        fprintf(fid,'%s\n',llfile);
        fprintf(fid,'%s\n',vars(i));
        fprintf(fid,'%s\n','m2,s2,n2,k2,k1,o1,p1,q1,s1,m4,mn4,ms4');
        fprintf(fid,'%s\n','AP');
        fprintf(fid,'%s\n','oce');
        fprintf(fid,'%s\n','0');
        fprintf(fid,'%s\n',[harmonics_prefix vars(i)]);
        fclose(fid);


        disp(['Extracting ' vars(i) ' Harmonics'])

        [s,w]=system('./extract_HC < otps_input');
        disp([' '])
        disp(['*******************************************'])
        disp(w)
        disp(['*******************************************'])
    end
end


    lon=ncread(gfile,'lon_rho');
    lat=ncread(gfile,'lat_rho');


    disp(['Mode parameter file used: ' model_file])




    % model_file='DATA\Model_AO';
    otps2frc_v7(gfile,base_date,pred_date,ofile,model_file,harmonics_prefix,'GulfOfGuinea')
    %
    % %
    disp('plotting')
    grd=roms_get_grid(gfile);

    param=ncread(ofile,'tide_Eamp');

    figure(3)
    clf

    pcolorjw(grd.lon_rho,grd.lat_rho,squeeze(param(:,:,1))')
    hold on
    %read_shapes_barnegat
    %shading flat
    hold on
    %plot(grd.lon_rho(grd.mask_rho==1),grd.lat_rho(grd.mask_rho==1),'.k')
    hold off

    colorbar


    param=ncread(ofile,'tide_Ephase');

    figure(4)
    clf
    colormap(jet)

    pcolorjw(grd.lon_rho,grd.lat_rho,squeeze(param(:,:,1))')
    hold on

    %shading flat
    hold on
    %plot(grd.lon_rho(grd.mask_rho==1),grd.lat_rho(grd.mask_rho==1),'.k')
    hold off

    colorbar