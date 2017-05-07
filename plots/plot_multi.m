close all
clear
clc

M = importdata('~/adaptive-batch-size/lqg/results/final/adabatch_est1_bound1__delta0_95_sample1.out',' ',1);
figure
hold on
color = 'b';
plot_scaledData
J_avg = sum(realJ.*batchsize)/sum(batchsize)

M = importdata('~/adaptive-batch-size/lqg/results/bernstein_test0',' ',1);
color = 'r--';
plot_scaledData
J_avg = sum(realJ.*batchsize)/sum(batchsize)

M = importdata('~/adaptive-batch-size/lqg/results/final/adabatch_est1_bound1__delta0_5_sample3.out',' ',1);
color = 'g';
plot_scaledData
J_avg = sum(realJ.*batchsize)/sum(batchsize)

legend('gpomdp .95', 'reinforce 0.95', 'gpomdp .5')
