iteration = M.data(:,1);
batchsize = M.data(:,2);
performance = M.data(:,4);
realJ = M.data(:,5);
J_scaled = scaleData(batchsize,performance,30000000,1000);
x = 1000*[1:length(J_scaled)];
plot(x,J_scaled,color)