close all
clear
clc

M = importdata('~/gradient_results/maxN01/adabatch_gpomdp_d1_0_max_30000000.out',' ',1);
figure
hold on
color = 'b';
plot_scaledData
J_avg = sum(realJ.*batchsize)/sum(batchsize)

M = importdata('~/gradient_results/maxN01/adabatch_gpomdp_d0_9_max_30000000.out',' ',1);
color = 'r';
plot_scaledData
J_avg = sum(realJ.*batchsize)/sum(batchsize)


M = importdata('~/gradient_results/maxN01/adabatch_gpomdp_d0_75_max_30000000.out',' ',1);
color = 'g';
plot_scaledData
J_avg = sum(realJ.*batchsize)/sum(batchsize)

M = importdata('~/gradient_results/maxN01/adabatch_gpomdp_d0_25_max_30000000.out',' ',1);
color = 'y';
plot_scaledData
J_avg = sum(realJ.*batchsize)/sum(batchsize)

legend('delta = 1.0','delta = 0.9','delta = 0.75','delta = 0.25')
