close all
clear
clc

figure
hold on

M = importdata('~/adaptive-batch-size/lqgnd/results/notheta/vector_ex.out',' ',1);
color = 'b-';
plot_scaledData2
J_avg = sum(realJ.*batchsize)/sum(batchsize)
improv = performance(2:length(performance)) - performance(1:length(performance)-1);
eff = sum(improv>0)/(length(performance)-1)

M = importdata('~/adaptive-batch-size/lqgnd/results/notheta/scalar_ex.out',' ',1);
color = 'r--';
plot_scaledData2
J_avg = sum(realJ.*batchsize)/sum(batchsize)
improv = performance(2:length(performance)) - performance(1:length(performance)-1);
eff = sum(improv>0)/(length(performance)-1)
