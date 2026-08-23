%% Step 2: Point Spread Function (PSF) and Optical Transfer Function (OTF)
% Analyzes impulse response and Fourier transfer characteristics.

clear; close all; clc;

fprintf('====================================================\n');
fprintf('  Step 2: PSF & Optical Transfer Function (OTF)\n');
fprintf('====================================================\n\n');

% 1. Load Data1 & Data2
if exist('../data/Data1.mat', 'file')
    d1 = load('../data/Data1.mat');
    d2 = load('../data/Data2.mat');
else
    d1 = load('Data1.mat');
    d2 = load('Data2.mat');
end

IR1 = d1.IR;
IR2 = d2.IR;
[N, M] = size(d1.Data);

fprintf('Data1 PSF size: %dx%d, sum: %.4f\n', size(IR1,1), size(IR1,2), sum(IR1(:)));
fprintf('Data2 PSF size: %dx%d, sum: %.4f\n\n', size(IR2,1), size(IR2,2), sum(IR2(:)));

% Pad and center PSFs to compute OTF
padded1 = zeros(N, M);
padded1(1:size(IR1,1), 1:size(IR1,2)) = IR1;
padded1 = circshift(padded1, -floor(size(IR1)/2));
H1 = fftshift(fft2(padded1));

padded2 = zeros(N, M);
padded2(1:size(IR2,1), 1:size(IR2,2)) = IR2;
padded2 = circshift(padded2, -floor(size(IR2)/2));
H2 = fftshift(fft2(padded2));

freq_x = (-M/2 : M/2-1) / M;
freq_y = (-N/2 : N/2-1) / N;
center_row = N/2 + 1;

% Figure 1: 3D Surface & OTF Maps
figure('Name', 'Step 2: PSF Analysis', 'Position', [80 80 1400 700]);

subplot(2, 3, 1);
surf(IR1); shading interp; colormap(gca, jet);
title('Data1: 3D Impulse Response (PSF)');
xlabel('x'); ylabel('y'); zlabel('h_1[n,m]'); view(45, 30);

subplot(2, 3, 2);
imagesc(freq_x, freq_y, log10(abs(H1) + eps));
colormap(gca, magma); colorbar; axis square;
xlabel('\nu_x'); ylabel('\nu_y');
title('Data1: log_{10}|H_1(\nu)| (Strictly > 0)');

subplot(2, 3, 3);
plot(freq_x, abs(H1(center_row, :)), 'b-', 'LineWidth', 2);
grid on; xlabel('\nu_x'); ylabel('|H_1(\nu_x, 0)|');
title('Data1: Horizontal Slice \nu_y=0');

subplot(2, 3, 4);
surf(IR2); shading interp; colormap(gca, jet);
title('Data2: 3D Impulse Response (PSF)');
xlabel('x'); ylabel('y'); zlabel('h_2[n,m]'); view(45, 30);

subplot(2, 3, 5);
imagesc(freq_x, freq_y, log10(abs(H2) + eps));
colormap(gca, magma); colorbar; axis square;
xlabel('\nu_x'); ylabel('\nu_y');
title('Data2: log_{10}|H_2(\nu)| (Zero Crossings)');

subplot(2, 3, 6);
plot(freq_x, abs(H2(center_row, :)), 'r-', 'LineWidth', 2);
grid on; xlabel('\nu_x'); ylabel('|H_2(\nu_x, 0)|');
title('Data2: Horizontal Slice \nu_y=0 (Zeros)');

fprintf('Theoretical Summary:\n');
fprintf('  - Data1 (Gaussian): H_1(nu) > 0 everywhere -> ill-conditioned but invertible.\n');
fprintf('  - Data2 (Box): H_2(nu) = sinc(7*pi*nu_x)*sinc(7*pi*nu_y) -> has true zeros.\n');
fprintf('  - Zero frequencies cause infinite noise amplification in unregularized inversion.\n\n');
