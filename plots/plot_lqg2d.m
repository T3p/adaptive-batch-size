close all
clear
clc

figure
hold on

M = importdata('~/adaptive-batch-size/lqg/results/final/adabatch_est0_bound0__delta0_95_sample1.out',' ',1);
color = 'b';
plot_scaledData
J_avg = sum(realJ.*batchsize)/sum(batchsize)
improv = performance(2:length(performance)) - performance(1:length(performance)-1);
eff = sum(improv>0)/(length(performance)-1)


M = importdata('~/adaptive-batch-size/lqg/results/final/adabatch_est0_bound0__delta0_75_sample1.out',' ',1);
color = 'r';
plot_scaledData
J_avg = sum(realJ.*batchsize)/sum(batchsize)
improv = performance(2:length(performance)) - performance(1:length(performance)-1);
eff = sum(improv>0)/(length(performance)-1)


M = importdata('~/adaptive-batch-size/lqg/results/final/adabatch_est0_bound0__delta0_5_sample1.out',' ',1);
color = 'g';
plot_scaledData
J_avg = sum(realJ.*batchsize)/sum(batchsize)
improv = performance(2:length(performance)) - performance(1:length(performance)-1);
eff = sum(improv>0)/(length(performance)-1)


M = importdata('~/adaptive-batch-size/lqg/results/final/adabatch_est0_bound0__delta0_25_sample1.out',' ',1);
color = 'c';
plot_scaledData
J_avg = sum(realJ.*batchsize)/sum(batchsize)
improv = performance(2:length(performance)) - performance(1:length(performance)-1);
eff = sum(improv>0)/(length(performance)-1)


M = importdata('~/adaptive-batch-size/lqg/results/final/adabatch_est0_bound0__delta0_05_sample1.out',' ',1);
color = 'y--';
plot_scaledData
J_avg = sum(realJ.*batchsize)/sum(batchsize)
improv = performance(2:length(performance)) - performance(1:length(performance)-1);
eff = sum(improv>0)/(length(performance)-1)

xlabel('Trajectory')
ylabel('Th. performance')
legend('adabatch', 'polgrad')
