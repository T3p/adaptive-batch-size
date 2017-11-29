close all
clear
clc

M = importdata('~/adaptive-batch-size/lqg/results/final/adabatch_est1_bound5__delta0_95_sample1.out',' ',1);
header = 'bern_emp_gpomdp_095';
save_scaledData
J_avg = sum(realJ.*batchsize)/sum(batchsize)
improv = performance(2:length(performance)) - performance(1:length(performance)-1);
eff = sum(improv>0)/(length(performance)-1)

M = importdata('~/adaptive-batch-size/lqg/results/final/adabatch_est1_bound4__delta0_95_sample1.out',' ',1);
header = 'hoeff_emp_gpomdp_095';
save_scaledData
J_avg = sum(realJ.*batchsize)/sum(batchsize)
improv = performance(2:length(performance)) - performance(1:length(performance)-1);
eff = sum(improv>0)/(length(performance)-1)

M = importdata('~/adaptive-batch-size/lqg/results/final/adabatch_est1_bound3__delta0_95_sample1.out',' ',1);
header = 'bernstein_gpomdp_095';
save_scaledData
J_avg = sum(realJ.*batchsize)/sum(batchsize)
improv = performance(2:length(performance)) - performance(1:length(performance)-1);
eff = sum(improv>0)/(length(performance)-1)

M = importdata('~/adaptive-batch-size/lqg/results/final/adabatch_est1_bound2__delta0_95_sample1.out',' ',1);
header = 'hoeffding_gpomdp_095';
save_scaledData
J_avg = sum(realJ.*batchsize)/sum(batchsize)
improv = performance(2:length(performance)) - performance(1:length(performance)-1);
eff = sum(improv>0)/(length(performance)-1)