close all
clear
clc

figure
hold on

xlabel('trajectory');
ylabel('expected performance');

M = importdata('~/adaptive-batch-size/processed_data/chebyshev_gpomdp_095.txt',' ',1);
plot(M.data(:,1),M.data(:,2),'b','DisplayName','G(PO)MDP \delta=0.95');


M = importdata('~/adaptive-batch-size/processed_data/chebyshev_gpomdp_075.txt',' ',1);
plot(M.data(:,1),M.data(:,2),'r','DisplayName','G(PO)MDP \delta=0.75');


M = importdata('~/adaptive-batch-size/processed_data/chebyshev_gpomdp_050.txt',' ',1)
plot(M.data(:,1),M.data(:,2),'g','DisplayName','G(PO)MDP \delta=0.5');


M = importdata('~/adaptive-batch-size/processed_data/chebyshev_gpomdp_025.txt',' ',1);
plot(M.data(:,1),M.data(:,2),'c','DisplayName','G(PO)MDP \delta=0.25');


M = importdata('~/adaptive-batch-size/processed_data/chebyshev_gpomdp_005.txt',' ',1);
plot(M.data(:,1),M.data(:,2),'y','DisplayName','G(PO)MDP \delta=0.05');


M = importdata('~/adaptive-batch-size/processed_data/chebyshev_reinforce_095.txt',' ',1);
plot(M.data(:,1),M.data(:,2),'b--','DisplayName','REINFORCE \delta=0.95');


M = importdata('~/adaptive-batch-size/processed_data/chebyshev_reinforce_075.txt',' ',1);
plot(M.data(:,1),M.data(:,2),'r--','DisplayName','REINFORCE \delta=0.75');


M = importdata('~/adaptive-batch-size/processed_data/chebyshev_reinforce_050.txt',' ',1);
plot(M.data(:,1),M.data(:,2),'g--','DisplayName','REINFORCE \delta=0.5');

legend('show','Location','southeast');