%% Step 1: 2D Fourier Spectral Analysis of Observed Images
% Analyzes spatial and frequency domain properties of Data1 and Data2.

clear; close all; clc;

fprintf('====================================================\n');
fprintf('  Step 1: Spectral Analysis of Observations\n');
fprintf('====================================================\n\n');

% 1. Load Data1
if exist('../data/Data1.mat', 'file')
    d1 = load('../data/Data1.mat');
else
    d1 = load('Data1.mat');
end
Data1 = d1.Data;
[N1, M1] = size(Data1);
fprintf('Data1: %dx%d pixels, dynamic range [%.2f, %.2f]\n', ...
        N1, M1, min(Data1(:)), max(Data1(:)));

% 2. Load Data2
if exist('../data/Data2.mat', 'file')
    d2 = load('../data/Data2.mat');
else
    d2 = load('Data2.mat');
end
Data2 = d2.Data;
[N2, M2] = size(Data2);
fprintf('Data2: %dx%d pixels, dynamic range [%.2f, %.2f]\n\n', ...
        N2, M2, min(Data2(:)), max(Data2(:)));

% Frequency coordinates
freq_x1 = (-M1/2 : M1/2-1) / M1;
freq_y1 = (-N1/2 : N1/2-1) / N1;

% Compute spectra
spec1 = fftshift(fft2(Data1));
mag1 = abs(spec1);
log_mag1 = log10(mag1 + 1);

spec2 = fftshift(fft2(Data2));
mag2 = abs(spec2);
log_mag2 = log10(mag2 + 1);

% Visualization
figure('Name', 'Step 1: Spectral Analysis', 'Position', [100 100 1200 600]);

subplot(2, 2, 1);
imshow(Data1, []); colorbar;
title('Data1: Observed Image (Spatial)');

subplot(2, 2, 2);
imagesc(freq_x1, freq_y1, log_mag1);
colormap(gca, jet); colorbar; axis square;
xlabel('\nu_x (cycles/pixel)'); ylabel('\nu_y (cycles/pixel)');
title('Data1: Spectrum log_{10}|Y_1(\nu)+1| (Isotropic)');

subplot(2, 2, 3);
imshow(Data2, []); colorbar;
title('Data2: Observed Image (Spatial)');

subplot(2, 2, 4);
imagesc(freq_x1, freq_y1, log_mag2);
colormap(gca, jet); colorbar; axis square;
xlabel('\nu_x (cycles/pixel)'); ylabel('\nu_y (cycles/pixel)');
title('Data2: Spectrum log_{10}|Y_2(\nu)+1| (Directional Nulls)');

fprintf('Key Observations:\n');
fprintf('  1. Data1 exhibits smooth, radially symmetric energy decay (Gaussian blur).\n');
fprintf('  2. Data2 exhibits anisotropic periodic dark striations (Box/motion null lines).\n');
fprintf('  3. In both cases, high-frequency signal is attenuated below noise floor.\n\n');
