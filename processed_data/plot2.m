close all
clear
clc

figure
hold on

xlabel('trajectory');
ylabel('expected performance');

M = importdata('~/adaptive-batch-size/processed_data/bern_emp_gpomdp_095.txt',' ',1);
plot(M.data(:,1),M.data(:,2),'r--','DisplayName','Bernstein (empirical range)');


M = importdata('~/adaptive-batch-size/processed_data/hoeff_emp_gpomdp_095.txt',' ',1);
plot(M.data(:,1),M.data(:,2),'g--','DisplayName','Hoeffding (empirical range)');


M = importdata('~/adaptive-batch-size/processed_data/bernstein_gpomdp_095.txt',' ',1);
plot(M.data(:,1),M.data(:,2),'r','DisplayName','Bernstein');


M = importdata('~/adaptive-batch-size/processed_data/chebyshev_gpomdp_095.txt',' ',1)
plot(M.data(:,1),M.data(:,2),'b','DisplayName','Chebyshev');


M = importdata('~/adaptive-batch-size/processed_data/hoeffding_gpomdp_095.txt',' ',1);
plot(M.data(:,1),M.data(:,2),'g','DisplayName','Hoeffding');

legend('show','Location','southeast');