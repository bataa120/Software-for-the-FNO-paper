% 2nd paper domain :
% 
% lon: 217 - 331 
%      [127.96]   -   [137.51]
% 
% lat: 244 - 310
%      [35.28]   -   [40.76]
clear;
load('u15_19930101_20241231.mat')
u_d=permute(u_d,[3 2 1]);
u_d=u_d(:,244:310,217:331);
save 'u15_d_128.mat' u_d
clear u_d

load('merc_lonlat.mat')

me_lat=me_lat(244:310,1);
me_lon=me_lon(217:331,1);

[lon_me,lat_me] = meshgrid(me_lon,me_lat);
lon_me = permute(lon_me,[2 1]);
lat_me = permute(lat_me,[2 1]);

load('v15_19930101_20241231.mat')
v_d=permute(v_d,[3 2 1]);
v_d=v_d(:,244:310,217:331);
save 'v15_d_128.mat' v_d
clear v_d

load('z_19930101_20241231.mat')
z_d=permute(z_d,[3 2 1]);
z_d=z_d(:,244:310,217:331);
save 'z_d_128.mat' z_d
clear z_d

load('interpolated_wind_u.mat')
u10=permute(u10,[3 2 1]);
u10=u10(:,244:310,217:331);
save 'u10_d_128.mat' u10
clear u10

load('interpolated_wind_v.mat')
v10=permute(v10,[3 2 1]);
v10=v10(:,244:310,217:331);
save 'v10_d_128.mat' v10
clear v10

clear;
load('t1_19930101_20241231.mat')
t1_d=permute(t1_d,[3 2 1]);
t1_d=t1_d(:,244:310,217:331);
save 't1_d_128.mat' t1_d
clear t1_d

clear;
load('t100_19930101_20241231.mat')
t100_d=permute(t100_d,[3 2 1]);
t100_d=t100_d(:,244:310,217:331);
save 't100_d_128.mat' t100_d
clear t100_d
disp('Finished!')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Slicing for paper plot
%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('u15_19930101_20241231.mat')
u=u_d(:,:,10092);

load('v15_19930101_20241231.mat')
v=v_d(:,:,10092);
load('z_19930101_20241231.mat')
z=z_d(:,:,10092);
load('t1_19930101_20241231.mat')
t1=t1_d(:,:,10092);
load('t100_19930101_20241231.mat')
t100=t100_d(:,:,10092);

