function [interval] = confidence_interval(x)
    SEM = std(x)/sqrt(length(x));              
    ts = tinv([0.05  0.95],length(x)-1);       
    interval = mean(x) + ts*SEM;                     
end

