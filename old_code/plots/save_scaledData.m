iteration = M.data(:,1);
batchsize = M.data(:,2);
performance = M.data(:,5);
realJ = M.data(:,6);
J_scaled = scaleData(batchsize,performance,30000000,1000);
x = 1000*[1:length(J_scaled)];
%plot(x,J_scaled,color)

fid=fopen(strcat('./',header,'.txt'),'w');
fprintf(fid, [ 'trajectory' ' ' 'performance' '\n']);
fprintf(fid, '%d %f\n',[x' J_scaled']');
fclose(fid);