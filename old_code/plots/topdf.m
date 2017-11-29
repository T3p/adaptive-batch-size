close all
clear
clc

h = hgload('cartpole.fig');
orient(h,'landscape');
print(gcf, '-dpdf', '~/adaptive-batch-size/plots/pdf/cartpole.pdf','-fillpage');

% h = hgload('stepsize.fig');
% print(gcf, '-dpdf', '~/adaptive-batch-size/plots/pdf/stepsize.pdf','-fillpage');

% h = hgload('compare_stepsize.fig');
% print(gcf, '-dpdf', '~/adaptive-batch-size/plots/pdf/compare_stepsize.pdf','-fillpage');

% h = hgload('compare_bounds.fig');
% print(gcf, '-dpdf', '~/adaptive-batch-size/plots/pdf/compare_bounds.pdf','-fillpage');

% h = hgload('chebyshev.fig');
% print(gcf, '-dpdf', '~/adaptive-batch-size/plots/pdf/chebyshev.pdf','-fillpage');

% h = hgload('lqg2d_batchsize.fig');
% orient(h,'landscape');
% print(gcf, '-dpdf', '~/adaptive-batch-size/plots/pdf/lqg2d_batchsize_landscape.pdf','-fillpage');

% h = hgload('lqg2d_batchsize.fig');
% print(gcf, '-dpdf', '~/adaptive-batch-size/plots/pdf/lqg2d_batchsize.pdf','-fillpage');

% h = hgload('lqg2d.fig');
% print(gcf, '-dpdf', '~/adaptive-batch-size/plots/pdf/lqg2d.pdf','-fillpage');

% h = hgload('batchsize_cheb.fig');
% print(gcf, '-dpdf', '~/adaptive-batch-size/plots/pdf/batchsize_cheb.pdf','-fillpage');

% h = hgload('batchsize_cheb.fig');
% orient(h,'landscape');
% print(gcf, '-dpdf', '~/adaptive-batch-size/plots/pdf/batchsize_cheb_landscape.pdf','-fillpage');

% h = hgload('batchsize_hoeff.fig');
% print(gcf, '-dpdf', '~/adaptive-batch-size/plots/pdf/batchsize_hoeff.pdf','-fillpage');
% 
% h = hgload('batchsize_bern.fig');
% print(gcf, '-dpdf', '~/adaptive-batch-size/plots/pdf/batchsize_bern.pdf','-fillpage');
% 
% h = hgload('batchsize_hoeff_emp.fig');
% print(gcf, '-dpdf', '~/adaptive-batch-size/plots/pdf/batchsize_hoeff_emp.pdf','-fillpage');
% 
% h = hgload('batchsize_bern_emp.fig');
% print(gcf, '-dpdf', '~/adaptive-batch-size/plots/pdf/batchsize_bern_emp.pdf','-fillpage');