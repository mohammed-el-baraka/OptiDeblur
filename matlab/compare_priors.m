%% Compare Regularization Priors: Tikhonov vs Gradient vs Laplacian
clear; close all; clc;

fprintf('====================================================\n');
fprintf('  Comparative Analysis of Regularization Operators\n');
fprintf('====================================================\n\n');

if exist('../data/Data1.mat', 'file')
    d = load('../data/Data1.mat');
else
    d = load('Data1.mat');
end

Data = d.Data;
IR = d.IR;
TrueImage = d.TrueImage;

priors = {'identity', 'gradient', 'laplacian'};
prior_names = {'Tikhonov (Identity)', '1st-Order Gradient', '2nd-Order Laplacian'};
colors = {'b', 'r', 'g'};

lambda_range = logspace(-6, 2, 80);
results = struct();

figure('Name', 'Prior Comparison', 'Position', [100 100 1200 700]);

for p = 1:length(priors)
    ptype = priors{p};
    pname = prior_names{p};
    fprintf('Testing %s prior...\n', pname);

    errs = zeros(size(lambda_range));
    for i = 1:length(lambda_range)
        restored = deconvolve_wiener(Data, IR, lambda_range(i), ptype, true);
        errs(i) = norm(restored(:) - TrueImage(:)) / norm(TrueImage(:));
    end

    [min_err, idx] = min(errs);
    opt_lambda = lambda_range(idx);
    fprintf('  -> Optimal \\lambda: %.4e, Min Error: %.4f\n', opt_lambda, min_err);

    results.(ptype).errors = errs;
    results.(ptype).opt_lambda = opt_lambda;
    results.(ptype).min_err = min_err;

    % Plot curve
    subplot(2, 3, [1, 2, 3]);
    semilogx(lambda_range, errs, colors{p}, 'LineWidth', 2.2, 'DisplayName', ...
             sprintf('%s (\\lambda^*=%.2e, Err=%.4f)', pname, opt_lambda, min_err));
    hold on;
    semilogx(opt_lambda, min_err, [colors{p} 'o'], 'MarkerSize', 8, 'LineWidth', 2, 'HandleVisibility', 'off');

    % Plot image
    subplot(2, 3, 3 + p);
    opt_restored = deconvolve_wiener(Data, IR, opt_lambda, ptype, true);
    imshow(opt_restored, []);
    title(sprintf('%s\n\\lambda^*=%.2e (Err: %.4f)', pname, opt_lambda, min_err), 'FontWeight', 'bold');
end

subplot(2, 3, [1, 2, 3]);
grid on;
xlabel('Regularization Parameter \lambda', 'FontSize', 12);
ylabel('Relative L_2 Error \Delta_2', 'FontSize', 12);
title('Error vs \lambda across Regularization Priors', 'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);

fprintf('\nDone.\n');
