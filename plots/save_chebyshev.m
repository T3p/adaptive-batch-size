close all
clear
clc

M = importdata('~/adaptive-batch-size/lqg/results/final/adabatch_est1_bound1__delta0_95_sample1.out',' ',1);
header = 'chebyshev_gpomdp_095';
save_scaledData
J_avg = sum(realJ.*batchsize)/sum(batchsize)
improv = performance(2:length(performance)) - performance(1:length(performance)-1);
eff = sum(improv>0)/(length(performance)-1)


M = importdata('~/adaptive-batch-size/lqg/results/final/adabatch_est1_bound1__delta0_75_sample1.out',' ',1);
header = 'chebyshev_gpomdp_075';
save_scaledData
J_avg = sum(realJ.*batchsize)/sum(batchsize)
improv = performance(2:length(performance)) - performance(1:length(performance)-1);
eff = sum(improv>0)/(length(performance)-1)


M = importdata('~/adaptive-batch-size/lqg/results/final/adabatch_est1_bound1__delta0_5_sample1.out',' ',1);
header = 'chebyshev_gpomdp_050';
save_scaledData
J_avg = sum(realJ.*batchsize)/sum(batchsize)
improv = performance(2:length(performance)) - performance(1:length(performance)-1);
eff = sum(improv>0)/(length(performance)-1)


M = importdata('~/adaptive-batch-size/lqg/results/final/adabatch_est1_bound1__delta0_25_sample1.out',' ',1);
header = 'chebyshev_gpomdp_025';
save_scaledData
J_avg = sum(realJ.*batchsize)/sum(batchsize)
improv = performance(2:length(performance)) - performance(1:length(performance)-1);
eff = sum(improv>0)/(length(performance)-1)


M = importdata('~/adaptive-batch-size/lqg/results/final/adabatch_est1_bound1__delta0_05_sample1.out',' ',1);
header = 'chebyshev_gpomdp_005';
save_scaledData
J_avg = sum(realJ.*batchsize)/sum(batchsize)
improv = performance(2:length(performance)) - performance(1:length(performance)-1);
eff = sum(improv>0)/(length(performance)-1)

M = importdata('~/adaptive-batch-size/lqg/results/final/adabatch_est0_bound0__delta0_95_sample1.out',' ',1);
header = 'chebyshev_reinforce_095';
save_scaledData
J_avg = sum(realJ.*batchsize)/sum(batchsize)
improv = performance(2:length(performance)) - performance(1:length(performance)-1);
eff = sum(improv>0)/(length(performance)-1)

M = importdata('~/adaptive-batch-size/lqg/results/final/adabatch_est0_bound0__delta0_75_sample1.out',' ',1);
header = 'chebyshev_reinforce_075';
save_scaledData
J_avg = sum(realJ.*batchsize)/sum(batchsize)
improv = performance(2:length(performance)) - performance(1:length(performance)-1);
eff = sum(improv>0)/(length(performance)-1)

M = importdata('~/adaptive-batch-size/lqg/results/final/adabatch_est0_bound0__delta0_5_sample1.out',' ',1);
header = 'chebyshev_reinforce_050';
save_scaledData
J_avg = sum(realJ.*batchsize)/sum(batchsize)
improv = performance(2:length(performance)) - performance(1:length(performance)-1);
eff = sum(improv>0)/(length(performance)-1)
