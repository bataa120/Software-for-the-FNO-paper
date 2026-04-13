load('merc_lonlat.mat')
load('era5_lonlat.mat')

load('u10_19930101_20221231.mat')
u10=permute(u10_d,[3,2,1]);
u10_d1=flipud(u10);
% 
% d1=u10_d1(:,100,end)
% 
% d2=u10(:,100,end)

[lon_me,lat_me] = meshgrid(me_lon,me_lat);


er_lat=flipud(er_lat);
[lon_er,lat_er] = meshgrid(er_lon,er_lat);

l=length(u10_d1(1,1,:));
u10=zeros([size(lat_me),l]);

for k=1:l
    var=squeeze(u10_d1(:,:,k));
    Vq = interp2(lon_er,lat_er,var,lon_me,lat_me);
    u10(:,:,k)=Vq;
end

save('interpolated_wind_u.mat','u10','-v7.3');
disp('Finished!')

load('v10_19930101_20221231.mat')
v10=permute(v10_d,[3,2,1]);
v10_d1=flipud(v10);
% 
% d1=v10_d1(:,100,end)
% 
% d2=v10(:,100,end)

load('merc_lonlat.mat')
load('era5_lonlat.mat')

[lon_me,lat_me] = meshgrid(me_lon,me_lat);


er_lat=flipud(er_lat);
[lon_er,lat_er] = meshgrid(er_lon,er_lat);

l=length(v10_d1(1,1,:));
v10=zeros([size(lat_me),l]);

for k=1:l
    var=squeeze(v10_d1(:,:,k));
    Vq = interp2(lon_er,lat_er,var,lon_me,lat_me);
    v10(:,:,k)=Vq;
end



save('interpolated_wind_v.mat','v10','-v7.3');
disp('Finished!')









