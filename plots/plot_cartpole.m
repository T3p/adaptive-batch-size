close all
clear
clc

figure
hold on

horizon = 1151669;

M = importdata('~/adaptive-batch-size/cartpole/results/notheta/tweak2_adabatch.out',' ',1);
color = 'b';
iteration = M.data(:,1);
batchsize = M.data(:,2);
alpha = M.data(:,3);
performance = M.data(:,4);
tick = 1;
J_scaled = scaleData(batchsize,performance,horizon,tick);
x = tick*[1:length(J_scaled)];
plot(x,J_scaled,color)
x_sum = cumsum(batchsize);
X = length(x_sum(x_sum<horizon))
J_avg = sum(performance(1:X).*batchsize(1:X))/sum(batchsize(1:X))
sum(batchsize)
improv = performance(2:X) - performance(1:X-1);
eff = sum(improv>0)/(X-1)

M = importdata('~/adaptive-batch-size/cartpole/results/notheta/long_tweak2_fixed.out',' ',1);
color = 'r';
iteration = M.data(:,1);
batchsize = M.data(:,2);
alpha = M.data(:,3);
performance = M.data(:,4);
tick = 1;
J_scaled = scaleData(batchsize,performance,horizon,tick);
x = tick*[1:length(J_scaled)];
plot(x,J_scaled,color)
x_sum = cumsum(batchsize);
X = length(x_sum(x_sum<horizon))
J_avg = sum(performance(1:X).*batchsize(1:X))/sum(batchsize(1:X))
sum(batchsize)
improv = performance(2:X) - performance(1:X-1);
eff = sum(improv>0)/(X-1)