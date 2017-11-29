function [interval] = confidence_interval(x)
    SEM = std(x)/sqrt(length(x));              
    ts = tinv([0.05  0.95],length(x)-1);       
    interval(1) = mean(x) + ts(1)*SEM;                     
    interval(2) = mean(x) + ts(2)*SEM;
end

