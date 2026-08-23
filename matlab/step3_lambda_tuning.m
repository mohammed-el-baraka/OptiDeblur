%% Step 3: Regularization Parameter Lambda Sweep and U-Curve Analysis
% Compares unregularized inversion vs Wiener-Hunt regularized restoration.

clear; close all; clc;

fprintf('====================================================\n');
fprintf('  Step 3: Lambda Parameter Tuning & U-Curves\n');
fprintf('====================================================\n\n');

% Dataset selection
choice = 1; % Change to 2 for Data2 or use interactive prompt if running in IDE
if ~exist('choice', 'var')
    choice = input('Select dataset (1 for Data1, 2 for Data2): ');
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

% 1. Simple Inverse Filter (lambda = 0)
fprintf('1. Testing Unregularized Inverse (lambda = 0)...\n');
restored_inv = deconvolve_wiener(Data, IR, 0.0, 'gradient', true);
err_inv = norm(restored_inv(:) - TrueImage(:)) / norm(TrueImage(:));
fprintf('   Naive Inverse Error: %.4f (Catastrophic noise amplification)\n\n', err_inv);

% 2. Logarithmic Lambda Sweep
fprintf('2. Sweeping lambda from 1e-8 to 1e4...\n');
lambda_range = logspace(-8, 4, 100);
errors = zeros(size(lambda_range));

for i = 1:length(lambda_range)
    restored = deconvolve_wiener(Data, IR, lambda_range(i), 'gradient', true);
    errors(i) = norm(restored(:) - TrueImage(:)) / norm(TrueImage(:));
end

[min_err, opt_idx] = min(errors);
opt_lambda = lambda_range(opt_idx);

fprintf('   Optimal lambda: %.6e\n', opt_lambda);
fprintf('   Minimum Relative L2 Error: %.6f\n\n', min_err);

% 3. Visualization
figure('Name', ['Step 3: ' name], 'Position', [100 100 1300 700]);

subplot(2, 3, 1);
imshow(TrueImage, []);
title('Ground Truth x^*', 'FontWeight', 'bold');

subplot(2, 3, 2);
imshow(Data, []);
title(['Observed (' name ')'], 'FontWeight', 'bold');

subplot(2, 3, 3);
imshow(restored_inv, []);
title(sprintf('Inverse (\\lambda=0)\nErr: %.2f', err_inv), 'FontWeight', 'bold', 'Color', 'r');

subplot(2, 3, 4);
restored_under = deconvolve_wiener(Data, IR, 1e-6, 'gradient', true);
imshow(restored_under, []);
title('Under-reg. (\lambda=10^{-6})');

subplot(2, 3, 5);
restored_opt = deconvolve_wiener(Data, IR, opt_lambda, 'gradient', true);
imshow(restored_opt, []);
title(sprintf('Optimal (\\lambda=%.2e)\nErr: %.4f', opt_lambda, min_err), 'FontWeight', 'bold', 'Color', 'b');

subplot(2, 3, 6);
restored_over = deconvolve_wiener(Data, IR, 100.0, 'gradient', true);
imshow(restored_over, []);
title('Over-reg. (\lambda=100)');

% Figure 2: U-Curve
figure('Name', 'Step 3: U-Curve', 'Position', [200 200 800 500]);
semilogx(lambda_range, errors, 'b-', 'LineWidth', 2.5);
hold on;
semilogx(opt_lambda, min_err, 'r*', 'MarkerSize', 15, 'LineWidth', 2);
grid on;
xlabel('Regularization Parameter \lambda', 'FontSize', 12);
ylabel('Relative L_2 Error \Delta_2', 'FontSize', 12);
title(sprintf('%s: Convex Regularization U-Curve', name), 'FontSize', 13, 'FontWeight', 'bold');
legend('Reconstruction Error \Delta_2(\lambda)', sprintf('Optimal \\lambda^* = %.2e', opt_lambda), 'Location', 'best');
