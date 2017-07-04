iteration = M.data(:,1);
batchsize = M.data(:,2);
performance = M.data(:,5);
realJ = M.data(:,6);
J_scaled = scaleData(batchsize,performance,30000000,1000);
x = 1000*[1:length(J_scaled)];
plot(x,J_scaled,color)
plot(iteration,performance,color)