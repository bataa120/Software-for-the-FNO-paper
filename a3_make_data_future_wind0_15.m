clear;close all;
load('/scratch1/bataa/process_newp_all/u15v15zu10v10t1t100_115_mean.mat')

ubar=mean(u_d,"all");
vbar=mean(v_d,"all");
zbar=mean(z_d,"all");

t1bar=mean(t1_d,"all");
t100bar=mean(t100_d,"all");

w=0;

% u_s=std(u_d,w,"all","omitnan");
% v_s=std(v_d,w,"all","omitnan");
% z_s=std(z_d,w,"all","omitnan");
% 
% t1_s=std(t1_d,w,"all","omitnan");
% t100_s=std(t100_d,w,"all","omitnan");




in=15;
out=10;
t=in+out;

[l,Nx,Ny] = size(u_d);

s=5;
o=in;

au=zeros(floor(l/s),7, Nx,Ny,t);
count = 1;

for k=1:s:l        
    if k+t-1+o<=l
        au(count,1,:,:,:) = permute(u_d(k:k+t-1,:,:),[2 3 1]);
        au(count,2,:,:,:) = permute(v_d(k:k+t-1,:,:),[2 3 1]);
        au(count,3,:,:,:) = permute(z_d(k:k+t-1,:,:),[2 3 1]);        
        au(count,4,:,:,:) = permute(u10(k:k+t-1,:,:),[2 3 1]);
        au(count,5,:,:,:) = permute(v10(k:k+t-1,:,:),[2 3 1]);
        au(count,6,:,:,:) = permute(u10(k+o:k+t-1+o,:,:),[2 3 1]);
        au(count,7,:,:,:) = permute(v10(k+o:k+t-1+o,:,:),[2 3 1]);
        count=count+1;
    end

end

au = au(1:count-1,:,:,:,:);
nme='u15v15zu10v10t1t100_115_s5_t25_wind_0_15.mat';

save(nme,"au")

disp('Finished! out='+string(o))





a1=mean(au(:,1,:,:,:),"all");
a2=mean(au(:,2,:,:,:),"all");
a3=mean(au(:,3,:,:,:),"all");
a4=mean(au(:,4,:,:,:),"all");
a5=mean(au(:,5,:,:,:),"all");
a6=mean(au(:,6,:,:,:),"all");
a7=mean(au(:,7,:,:,:),"all");



