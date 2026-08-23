%% Wiener-Hunt Image Deconvolution - Main Interactive Demo
clear; close all; clc;

fprintf('=================================================================\n');
fprintf('       WIENER-HUNT IMAGE DECONVOLUTION BENCHMARK DEMO            \n');
fprintf('=================================================================\n\n');

choice = input('Select dataset to restore (1: Gaussian blur, 2: Box blur): ');
if isempty(choice) || choice ~= 2
    choice = 1;
end

if choice == 1
    filename = 'Data1.mat';
    name = 'Data1 (Gaussian Blur)';
else
    filename = 'Data2.mat';
    name = 'Data2 (Box Blur)';
end

if exist(['../data/' filename], 'file')
    d = load(['../data/' filename]);
else
    d = load(filename);
end

Data = d.Data;
IR = d.IR;
TrueImage = d.TrueImage;

fprintf('\nLoaded %s (%dx%d pixels)\n', name, size(Data,1), size(Data,2));
fprintf('PSF size: %dx%d\n\n', size(IR,1), size(IR,2));

% Parameter sweep
lambda_range = logspace(-7, 3, 50);
errors = zeros(size(lambda_range));

fprintf('Searching for optimal regularization parameter...\n');
for i = 1:length(lambda_range)
    res = deconvolve_wiener(Data, IR, lambda_range(i), 'gradient', true);
    errors(i) = norm(res(:) - TrueImage(:)) / norm(TrueImage(:));
end

[min_err, opt_idx] = min(errors);
best_lambda = lambda_range(opt_idx);
best_restored = deconvolve_wiener(Data, IR, best_lambda, 'gradient', true);

fprintf('Optimal \\lambda : %.4e\n', best_lambda);
fprintf('Minimum Error   : %.4f\n\n', min_err);

% Visual Display
figure('Name', 'Wiener-Hunt Deconvolution Results', 'Position', [100 100 1400 500]);

subplot(1, 3, 1);
imshow(TrueImage, []);
title('Ground Truth x^*', 'FontWeight', 'bold', 'FontSize', 12);

subplot(1, 3, 2);
imshow(Data, []);
title(sprintf('Observed (%s)', name), 'FontWeight', 'bold', 'FontSize', 12);

subplot(1, 3, 3);
imshow(best_restored, []);
title(sprintf('Restored (\\lambda=%.2e, Err=%.4f)', best_lambda, min_err), ...
      'FontWeight', 'bold', 'FontSize', 12, 'Color', 'b');

fprintf('Demo complete.\n');
