%% Step 4: Multi-Norm Error Evaluation and Metric Concordance
% Evaluates L1, L2, and L-infinity distance norms across regularization levels.

clear; close all; clc;

fprintf('====================================================\n');
fprintf('  Step 4: Distance Metrics Concordance\n');
fprintf('====================================================\n\n');

if exist('../data/Data2.mat', 'file')
    d = load('../data/Data2.mat');
else
    d = load('Data2.mat');
end

Data = d.Data;
IR = d.IR;
TrueImage = d.TrueImage;

lambda_vals = logspace(-6, 3, 100);
n = length(lambda_vals);

delta_L2 = zeros(1, n);
delta_L1 = zeros(1, n);
delta_Linf = zeros(1, n);

fprintf('Evaluating %d lambda values...\n', n);
for i = 1:n
    restored = deconvolve_wiener(Data, IR, lambda_vals(i), 'gradient', true);
    diff = restored - TrueImage;

    delta_L2(i) = norm(diff(:), 2) / norm(TrueImage(:), 2);
    delta_L1(i) = norm(diff(:), 1) / norm(TrueImage(:), 1);
    delta_Linf(i) = max(abs(diff(:))) / max(abs(TrueImage(:)));
end

[min_L2, idx_L2] = min(delta_L2);
[min_L1, idx_L1] = min(delta_L1);
[min_Linf, idx_Linf] = min(delta_Linf);

lambda_L2 = lambda_vals(idx_L2);
lambda_L1 = lambda_vals(idx_L1);
lambda_Linf = lambda_vals(idx_Linf);

fprintf('\nOptimal Parameters Found:\n');
fprintf('  - L2 Norm    (\\Delta_2):    \\lambda^* = %.4e (Min error = %.4f)\n', lambda_L2, min_L2);
fprintf('  - L1 Norm    (\\Delta_1):    \\lambda^* = %.4e (Min error = %.4f)\n', lambda_L1, min_L1);
fprintf('  - L-inf Norm (\\Delta_inf):  \\lambda^* = %.4e (Min error = %.4f)\n\n', lambda_Linf, min_Linf);

% Save results structure
results.lambda_vals = lambda_vals;
results.delta_L2 = delta_L2;
results.delta_L1 = delta_L1;
results.delta_Linf = delta_Linf;
results.optimal_L2 = lambda_L2;
results.optimal_L1 = lambda_L1;
results.optimal_Linf = lambda_Linf;

save('distance_results.mat', 'results');
fprintf('Saved results to distance_results.mat\n\n');

% Figure: Distance Metrics Comparison
figure('Name', 'Step 4: Metric Concordance', 'Position', [150 150 1000 600]);
semilogx(lambda_vals, delta_L2, 'b-', 'LineWidth', 2.2, 'DisplayName', '\Delta_2 (L_2)');
hold on;
semilogx(lambda_vals, delta_L1, 'r--', 'LineWidth', 2.2, 'DisplayName', '\Delta_1 (L_1)');
semilogx(lambda_vals, delta_Linf, 'g-.', 'LineWidth', 2.2, 'DisplayName', '\Delta_\infty (L_\infty)');

semilogx(lambda_L2, min_L2, 'bo', 'MarkerSize', 10, 'LineWidth', 2, 'HandleVisibility', 'off');
semilogx(lambda_L1, min_L1, 'ro', 'MarkerSize', 10, 'LineWidth', 2, 'HandleVisibility', 'off');
semilogx(lambda_Linf, min_Linf, 'go', 'MarkerSize', 10, 'LineWidth', 2, 'HandleVisibility', 'off');

grid on;
xlabel('Regularization Parameter \lambda', 'FontSize', 12);
ylabel('Normalized Distance Metric', 'FontSize', 12);
title('Metric Concordance across Regularization Regimes', 'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'north', 'FontSize', 11);
