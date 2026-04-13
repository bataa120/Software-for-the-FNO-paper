clear; close all;
load('u15_d_128.mat');
load('v15_d_128.mat');
load('z_d_128.mat');
load('u10_d_128.mat');
load('v10_d_128.mat');

load('t1_d_128.mat');
load('t100_d_128.mat');

d=size(u10);
u_d=u_d(1:d(1),:,:);
v_d=v_d(1:d(1),:,:);
z_d=z_d(1:d(1),:,:);
t1_d=t1_d(1:d(1),:,:);
t100_d=t100_d(1:d(1),:,:);




mask_u=isnan(u_d);
mask_v=isnan(v_d);
mask_z=isnan(z_d);
mask_t1=isnan(t1_d);
mask_t100=isnan(t100_d);


mask=mask_u|mask_v|mask_z|mask_t1|mask_t100;

u_d(mask)=0;
v_d(mask)=0;

u10(mask)=0;
v10(mask)=0;

m_u=mean(u_d(~mask),"all");
m_v=mean(v_d(~mask),"all");



m_z=mean(z_d(~mask),"all");
z_d(mask)=m_z;

m_t1=mean(t1_d(~mask),"all");
t1_d(mask)=m_t1;

m_t100=mean(t100_d(~mask),"all");
t100_d(mask)=m_t100;


w=0;
u_s=std(u_d(~mask),w,"all","omitnan");
v_s=std(v_d(~mask),w,"all","omitnan");
z_s=std(z_d(~mask),w,"all","omitnan");

t1_s=std(t1_d(~mask),w,"all","omitnan");
t100_s=std(t100_d(~mask),w,"all","omitnan");



%save('uvzu10v10t1t100_115_mean.mat', 'u_d','v_d','z_d','u10','v10','t1_d','t100_d');