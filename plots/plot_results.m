close all
clear
clc

M = importdata('~/gradient_results/adabatch_gpomdp_d0_25_max_30000000.out',' ',1);
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
T = 1000;
plot(iteration(t:T),realJ(t:T))

J_avg = sum(realJ.*batchsize)/sum(batchsize)
sum(batchsize)
