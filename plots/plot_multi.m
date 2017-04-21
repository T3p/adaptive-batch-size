close all
clear
clc

M = importdata('~/adaptive-batch-size/lqg/results/adabatch_gpomdp_d1_0_max30.out',' ',1);
figure
hold on
color = 'b';
plot_scaledData
J_avg = sum(realJ.*batchsize)/sum(batchsize)

M = importdata('~/adaptive-batch-size/lqg/results/adabatch_gpomdp_d0_9_max30.out',' ',1);
color = 'r';
plot_scaledData
J_avg = sum(realJ.*batchsize)/sum(batchsize)


M = importdata('~/adaptive-batch-size/lqg/results/adabatch_gpomdp_d0_8_max30.out',' ',1);
color = 'g';
plot_scaledData
J_avg = sum(realJ.*batchsize)/sum(batchsize)

M = importdata('~/adaptive-batch-size/lqg/results/adabatch_gpomdp_d0_75_max30.out',' ',1);
color = 'y';
plot_scaledData
J_avg = sum(realJ.*batchsize)/sum(batchsize)

M = importdata('~/adaptive-batch-size/lqg/results/adabatch_gpomdp_d0_95_hoeffding.out',' ',1);
b = M.data(:,2);
T = length(b(b<30000000));
M.data = M.data(1:T,:);
color = 'm';
plot_scaledData
J_avg = sum(realJ.*batchsize)/sum(batchsize)

legend('delta = 1.0','delta = 0.9','delta = 0.8','delta = 0.75','hoeffding, delta = 0.95')
