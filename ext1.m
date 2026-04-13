clear; close all;

input_path = ['/scratch4/COMMON/MERCATOR/daily/GLORYS_REANALYSIS_2019.nc'];

me_lon = ncread([input_path],'longitude');
me_lat = ncread([input_path],'latitude');

yr = {'1993','1994','1995','1996','1997','1998','1999','2000','2001','2002'...
      ,'2003','2004','2005','2006','2007','2008','2009','2010','2011','2012'...
      ,'2013','2014','2015','2016','2017','2018','2019','2020','2021-06-30.nc','2021-12-31.nc'...
      ,'2022','2023', '2024'};
  
  

u_d=zeros(661,541,(2024-1993+1)*366);
v_d=zeros(661,541,(2024-1993+1)*366);

% t100_d=zeros(661,541,(2023-1993+1)*366);
% t1_d=zeros(661,541,(2023-1993+1)*366);
% z_d=zeros(661,541,(2023-1993+1)*366);

c=1;

for year=1:33
    iyear=yr{1,year};
    if isequal(iyear,'2021-06-30.nc')
        input_path = ['/scratch4/COMMON/MERCATOR/daily/GLORYS_REANALYSIS_',iyear];
    elseif isequal(iyear,'2021-12-31.nc')
        input_path = ['/scratch4/COMMON/MERCATOR/daily/GLORYS_REANALYSIS_',iyear];
    else
        input_path = ['/scratch4/COMMON/MERCATOR/daily/GLORYS_REANALYSIS_',iyear,'.nc'];
    end
%     
    t = ncread([input_path],'time');
    t=length(t);
    depth = ncread([input_path],'depth');
    fprintf('depth(1) = %.3f, depth(11) = %.3f, depth(22) = %.3f\n', depth(1), depth(11), depth(22));

    start=[1 1 11 1];
    count=[Inf Inf 1 Inf];
    u = squeeze(ncread([input_path],'uo',start,count));
    v = squeeze(ncread([input_path],'vo',start,count));
    
    u_d(:,:,c:c+t-1) = u;
    v_d(:,:,c:c+t-1) = v;
%     
%     start=[1 1 1 1];
%     count=[Inf Inf 1 Inf];
%     tem1 = squeeze(ncread([input_path],'thetao',start,count));
%     t1_d(:,:,c:c+t-1) = tem1;
    
%     start=[1 1 22 1];
%     count=[Inf Inf 1 Inf];
%     tem100 = squeeze(ncread([input_path],'thetao',start,count));
%     
%     t100_d(:,:,c:c+t-1) = tem100;
%     
%     z = ncread([input_path],'zos');
%     
%     z_d(:,:,c:c+t-1) = z;
    
    
    c=c+t;
    
    
    
% 
%     start=[1 1 1 1];
%     count=[Inf Inf 1 Inf];
%     tem0 = ncread([input_path],'thetao',start,count);
% 

% 
%     z = ncread([input_path],'zos');
    
    
    
end
c=c-1

u_d=u_d(:,:,1:c);
v_d=v_d(:,:,1:c);

% u_d=permute(u_d,[3 1 2]);
% v_d=permute(v_d,[3 1 2]);

% t1_d=t1_d(:,:,1:c);
% t100_d=t100_d(:,:,1:c);
% z_d=z_d(:,:,1:c);
% 
% t1_d=permute(t1_d,[3 1 2]);
% t100_d=permute(t100_d,[3 1 2]);
% z_d=permute(z_d,[3 1 2]);

save('u15_19930101_20241231.mat','u_d','-v7.3');
save('v15_19930101_20241231.mat','v_d','-v7.3');


% save('t1_19930101_20231231.mat','t1_d','-v7.3');
% save('t100_19930101_20231231.mat','t100_d','-v7.3');
% save('z_19930101_20231231.mat','z_d','-v7.3');


disp('Finished');
