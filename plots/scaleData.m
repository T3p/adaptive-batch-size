function [z] = scaleData(x,y,sup,step)
    x_sum = cumsum(x);
    for i = [0:sup/step]
        n = step*i;
        j = length(x_sum(x_sum<n));
        if j==0
            j = 1;
        end
        z(i+1) = y(j);
    end
end

