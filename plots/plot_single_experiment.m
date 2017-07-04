close all
clear
clc

M = importdata('~/adaptive-batch-size/lqgnd/results/notheta/adabatch_theta0_sample1.out',' ',1);
alphas = M.data(:,3);
M.data(:,3) = [];
iteration = M.data(:,1);
batchsize = M.data(:,2);
performance = M.data(:,3);
realJ = M.data(:,4);
delta_J = performance(2:length(performance)) - performance(1:length(performance)-1);
eff = length(delta_J>0)/length(delta_J)

t = 1;
T = length(batchsize(cumsum(batchsize)<30000000));

figure
plot(iteration(t:T),batchsize(t:T))
title('Batch-size')
xlabel('iteration')
ylabel('N')

% figure
% plot(iteration(t:T),performance(t:T))
% title('Performance')
% xlabel('iteration')
% ylabel('J')
% hold on
%plot(iteration(t:T),realJ(t:T),'r')

% figure 
% plot(iteration(t:T),realJ(t:T))
% title('Online Performance')

J_avg = sum(realJ(t:T).*batchsize(t:T))/sum(batchsize(t:T))
sum(batchsize(t:T))