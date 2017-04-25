close all
clear
clc

M = importdata('~/adaptive-batch-size/lqg/results/adabatch_gpomdp_d0_95_hoeffding_const.out',' ',1);
iteration = M.data(:,1);
batchsize = M.data(:,2);
performance = M.data(:,4);
realJ = M.data(:,5);

t = 1;
T = length(batchsize(cumsum(batchsize)<30000000));

figure
plot(iteration(t:T),batchsize(t:T))
title('Batch-size')
xlabel('iteration')
ylabel('N')

figure
plot(iteration(t:T),performance(t:T))
title('Performance')
xlabel('iteration')
ylabel('J')
hold on
plot(iteration(t:T),realJ(t:T),'r')

figure 
plot(iteration(t:T),realJ(t:T))
title('Online Performance')

J_avg = sum(realJ(t:T).*batchsize(t:T))/sum(batchsize(t:T))
sum(batchsize(t:T))
