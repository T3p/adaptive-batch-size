close all
clear
clc

M = importdata('~/adaptive-batch-size/lqgnd/results/theta/polgrad_theta05_N5000_alpha6_sample1.out',' ',1);
x = M.data(:,1);
y = M.data(:,4);
c = linspace(1,255,length(x));
figure
scatter(x,y,1,c)