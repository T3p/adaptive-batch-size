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

M = importdata('~/adaptive-batch-size/lqg/results/adabatch_gpomdp_d0_95_hoeffding_theo.out',' ',1);
color = 'b--';
plot_scaledData
J_avg = sum(realJ.*batchsize)/sum(batchsize)

M = importdata('~/adaptive-batch-size/lqg/results/adabatch_bernstein2_2_1000_99.out',' ',1);
M.data(:,3) = [];
color = 'r--';
plot_scaledData
J_avg = sum(realJ.*batchsize)/sum(batchsize)

legend('delta = 1.0','delta = 0.9','delta = 0.8','delta = 0.75', 'hoeff delta = 0.95','bernstein delta = 0.99')
