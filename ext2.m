clear; close all;

input_path = ['/scratch4/COMMON/ERA5/2000.nc'];

er_lon = ncread([input_path],'longitude');
er_lat = ncread([input_path],'latitude');



t = ncread([input_path],'time');
t=length(t);
d=t/8;
u = ncread([input_path],'u10');
sz=size(u);
u10=zeros(sz(1),sz(2),d);
for k=1:d
    u10(:,:,k)=mean(u(:,:,(d-1)*8+1:d*8),3);
end

size(u10)

yr = {'1993','1994','1995','1996','1997','1998','1999','2000','2001','2002'...
      ,'2003','2004','2005','2006','2007','2008','2009','2010','2011','2012'...
      ,'2013','2014','2015','2016','2017','2018','2019','2020','2021','2022'...
      ,'2023','2024'};

u10_d=zeros(241,221,(2024-1993+1)*366);
v10_d=zeros(241,221,(2024-1993+1)*366);

c=1;

for year=1:32
    iyear=yr{1,year};
    input_path = ['/scratch4/COMMON/ERA5/',iyear,'.nc'];
        
    t = ncread([input_path],'time');
    t=length(t);
    d=t/8;
    
    u = ncread([input_path],'u10');
    sz=size(u);
    u10=zeros(sz(1),sz(2),d);
    for k=1:d
        u10(:,:,k)=mean(u(:,:,(k-1)*8+1:k*8),3);
    end
    
    u10_d(:,:,c:c+d-1) = u10;
    
    v = ncread([input_path],'v10');
    sz=size(v);
    v10=zeros(sz(1),sz(2),d);
    for k=1:d
        v10(:,:,k)=mean(v(:,:,(k-1)*8+1:k*8),3);
    end
    
    v10_d(:,:,c:c+d-1) = v10;    
    
    c=c+d;    
    
end
c=c-1

u10_d=u10_d(:,:,1:c);
v10_d=v10_d(:,:,1:c);

u10_d=permute(u10_d,[3 1 2]);
v10_d=permute(v10_d,[3 1 2]);

save('u10_19930101_20241231.mat','u10_d','-v7.3');
save('v10_19930101_20241231.mat','v10_d','-v7.3');

disp('Finished');
