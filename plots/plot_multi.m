close all
clear
clc

M = importdata('~/adaptive-batch-size/lqg/results/adabatch_gpomdp_d1_0_max_30000000_e4.out',' ',1);
figure
hold on
color = 'b';
plot_scaledData
J_avg = sum(realJ.*batchsize)/sum(batchsize)

M = importdata('~/adaptive-batch-size/lqg/results/adabatch_gpomdp_d0_9_max_30000000_e4.out',' ',1);
color = 'r';
plot_scaledData
J_avg = sum(realJ.*batchsize)/sum(batchsize)


M = importdata('~/adaptive-batch-size/lqg/results/adabatch_gpomdp_d0_8_max_30000000_e4.out',' ',1);
color = 'g';
plot_scaledData
J_avg = sum(realJ.*batchsize)/sum(batchsize)

M = importdata('~/adaptive-batch-size/lqg/results/adabatch_gpomdp_d0_75_max_30000000_e4.out',' ',1);
color = 'y';
plot_scaledData
J_avg = sum(realJ.*batchsize)/sum(batchsize)

legend('delta = 1.0','delta = 0.9','delta = 0.8','delta = 0.75')
