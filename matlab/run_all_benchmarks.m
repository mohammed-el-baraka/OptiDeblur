%% Automated Benchmark Script (Non-Interactive)
% Executes full analysis on Data1 and Data2 and prints summary table.

clear; close all; clc;

fprintf('=================================================================\n');
fprintf('       WIENER-HUNT DECONVOLUTION AUTOMATED BENCHMARKS             \n');
fprintf('=================================================================\n\n');

datasets = {'Data1', 'Data2'};
data_files = {'Data1.mat', 'Data2.mat'};
blur_types = {'Gaussian Blur', '7x7 Box Blur'};

priors = {'identity', 'gradient', 'laplacian'};
prior_names = {'Identity (Tikhonov)', '1st-Order Gradient', '2nd-Order Laplacian'};

lambda_grid = logspace(-7, 3, 100);

fprintf('%-8s | %-20s | %-12s | %-12s\n', 'Dataset', 'Prior Operator', 'Opt Lambda', 'Min L2 Error');
fprintf('-----------------------------------------------------------------\n');

for d = 1:length(datasets)
    if exist(['../data/' data_files{d}], 'file')
        mat = load(['../data/' data_files{d}]);
    else
        mat = load(data_files{d});
    end

    Data = mat.Data;
    IR = mat.IR;
    TrueImage = mat.TrueImage;

    for p = 1:length(priors)
        ptype = priors{p};
        errs = zeros(size(lambda_grid));

        for i = 1:length(lambda_grid)
            res = deconvolve_wiener(Data, IR, lambda_grid(i), ptype, true);
            errs(i) = norm(res(:) - TrueImage(:)) / norm(TrueImage(:));
        end

        [min_err, opt_idx] = min(errs);
        opt_lam = lambda_grid(opt_idx);

        fprintf('%-8s | %-20s | %-12.4e | %-12.4f\n', ...
                datasets{d}, prior_names{p}, opt_lam, min_err);
    end
    fprintf('-----------------------------------------------------------------\n');
end

fprintf('\nAll benchmarks completed successfully.\n');
