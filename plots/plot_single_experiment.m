close all
clear
clc

M = importdata('~/adaptive-batch-size/lqg/results/adabatch_gpomdp_d0_95_max_30000000_unbiased_1.out',' ',1);
iteration = M.data(:,1);
batchsize = M.data(:,2);
performance = M.data(:,4);
realJ = M.data(:,5);

figure
plot(iteration,batchsize)
title('Batch-size')
xlabel('iteration')
ylabel('N')

figure
plot(iteration,performance)
title('Performance')
xlabel('iteration')
ylabel('J')
hold on
plot(iteration,realJ,'r')

figure 
t = 1;
T = length(iteration);
plot(iteration(t:T),realJ(t:T))

J_avg = sum(realJ.*batchsize)/sum(batchsize)
sum(batchsize)
