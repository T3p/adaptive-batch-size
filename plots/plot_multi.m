close all
clear
clc

M = importdata('~/adaptive-batch-size/lqg/results/final/adabatch_est1_bound1__delta0_95_sample1.out',' ',1);
figure
hold on
color = 'b';
plot_scaledData
J_avg = sum(realJ.*batchsize)/sum(batchsize)

M = importdata('~/adaptive-batch-size/lqg/results/final/adabatch_est1_bound2__delta0_95_sample1.out',' ',1);
color = 'r--';
plot_scaledData
J_avg = sum(realJ.*batchsize)/sum(batchsize)

xlabel('Trajectory')
ylabel('Th. performance')
legend('chebyshev .95', 'bernstein .95')
