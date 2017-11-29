close all
clear
clc

N_max = 30000000;
N_rate = 1000;
filename = '~/adaptive-batch-size/lqg/results/final/adabatch_est%d_bound%d__delta0_%d_sample%d.out';
figure
hold on
plots = [];

delta = 95;
est = 1;
bound = 1;
for est = [1]
    for delta = [95]
        for bound = [1 2 3]
            T = N_max;
            for i = 1:5
                M = importdata(sprintf(filename,est,bound,delta,i),' ',1);
                T = min(T,length(M.data(:,1)));
            end

            for i = 1:5
                M = importdata(sprintf(filename,est,bound,delta,i),' ',1);
                iteration = M.data(1:T,1);
                batchsize = M.data(1:T,2);
                cum_batchsize = cumsum(batchsize);
                J(i,:) = M.data(1:T,5);
                J_sample = M.data(1:T,6);
                J_avg(i) = sum(J_sample.*batchsize)/sum(batchsize);

                for j = 1:N_max/N_rate
                    N = j*N_rate;
                    J_scaled(i,j) = J(i,length(cum_batchsize(cum_batchsize<N)));
                end
            end

            J_mean = mean(J_scaled);

            plots = [plots,plot(N_rate*[1:length(J_mean)],J_mean,'b')];

            Upsilon = mean(J_avg)
            Upsilon_in = mean(J_avg)-tinv([0.05,0.95],length(J_avg)-1)*std(J_avg)/sqrt(length(J_avg))
            clear J;
            clear J_avg;
            clear J_scaled;
        end
    end
end