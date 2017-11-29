close all
clear
clc

figure
hold on

M = importdata('~/adaptive-batch-size/lqgnd/results/notheta/vector_ex.out',' ',1);
color = 'b-';
plot(M.data(:,1),M.data(:,3),color);
performance = M.data(:,4);
delta_J = performance(2:length(performance)) - performance(1:length(performance)-1);
eff = length(delta_J>0)/length(delta_J)

M = importdata('~/adaptive-batch-size/lqgnd/results/notheta/scalar_ex.out',' ',1);
color = 'r--';
plot(M.data(:,1),M.data(:,3),color);
performance = M.data(:,4);
delta_J = performance(2:length(performance)) - performance(1:length(performance)-1);
eff = length(delta_J>0)/length(delta_J)
