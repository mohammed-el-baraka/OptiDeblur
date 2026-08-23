function output = apply_edge_taper(input_img, psf, taper_width)
% APPLY_EDGE_TAPER Blends image boundaries with mean value using cosine window.
%   Mitigates boundary wrap-around ringing artifacts in FFT deconvolution.
%
% Syntax:
%   output = apply_edge_taper(input_img, psf)
%   output = apply_edge_taper(input_img, psf, taper_width)

    [m, n] = size(input_img);
    [pm, pn] = size(psf);

    if nargin < 3 || isempty(taper_width)
        width_m = min(round(pm * 2), round(m / 10));
        width_n = min(round(pn * 2), round(n / 10));
    else
        width_m = taper_width(1);
        width_n = taper_width(2);
    end

    alpha = ones(m, n);

    for i = 1:width_m
        weight = 0.5 * (1 - cos(pi * i / width_m));
        alpha(i, :) = min(alpha(i, :), weight);
        alpha(m - i + 1, :) = min(alpha(m - i + 1, :), weight);
    end

    for j = 1:width_n
        weight = 0.5 * (1 - cos(pi * j / width_n));
        alpha(:, j) = min(alpha(:, j), weight);
        alpha(:, n - j + 1) = min(alpha(:, n - j + 1), weight);
    end

    mean_val = mean(input_img(:));
    output = mean_val + alpha .* (input_img - mean_val);
end
