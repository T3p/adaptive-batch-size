close all
clear
clc

figure
hold on

T = 1000000;
filename = '~/adaptive-batch-size/lqg/results/final/adabatch_est1_bound1__delta0_%d_sample%d.out';

delta = 5;

for i = 1:5
    M = importdata(sprintf(filename,delta,i),' ',1);
    T = min(T,length(M.data(:,1)));
end

for i = 1:5
    M = importdata(sprintf(filename,delta,i),' ',1);
    iterations(i,:) = M.data(1:T,1);
    batchsizes(i,:) = M.data(1:T,2);
    thetas(i,:) = M.data(1:T,3);
    alphas(i,:) = M.data(1:T,4);
    Js(i,:) = M.data(1:T,5);
    J_samples(i,:) = M.data(1:T,6);
    J_avg(i) = sum(J_samples(i,:).*batchsizes(i,:))/sum(batchsizes(i,:));
end

toplot = Js;

for k = 1:T
    x = toplot(:,k);
    SEM = std(x)/sqrt(length(x));              
    ts = tinv([0.05  0.95],length(x)-1);       
    mu(k) = mean(x);
    upper(k) = mean(x) + ts(1)*SEM;                     
    lower(k) = mean(x) + ts(2)*SEM;
end

plot(1:T,mu,'b')
plot(1:T,upper,'r--')
plot(1:T,lower,'r--')
title('gpomdp, chebyshev, delta=0.95, 5 runs')

Upsilon = mean(J_avg)
Upsilon_in = mean(J_avg)-tinv([0.05,0.95],length(J_avg)-1)*std(J_avg)/sqrt(length(J_avg))