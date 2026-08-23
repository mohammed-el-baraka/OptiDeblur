function [restored, transfer_fn] = deconvolve_wiener(blurred, psf, lambda, reg_type, center_psf)
% DECONVOLVE_WIENER Regularized Wiener-Hunt image deconvolution.
%
% Syntax:
%   [restored, transfer_fn] = deconvolve_wiener(blurred, psf, lambda)
%   [restored, transfer_fn] = deconvolve_wiener(blurred, psf, lambda, reg_type)
%   [restored, transfer_fn] = deconvolve_wiener(blurred, psf, lambda, reg_type, center_psf)
%
% Inputs:
%   blurred    - Observed 2D image matrix (double)
%   psf        - Point Spread Function / impulse response matrix
%   lambda     - Regularization trade-off parameter (scalar >= 0)
%   reg_type   - Regularization operator: 'gradient' (default), 'identity', 'laplacian'
%   center_psf - Boolean flag: circularly shift PSF center to (1,1) (default: true)
%
% Outputs:
%   restored    - Reconstructed spatial image
%   transfer_fn - 2D complex frequency transfer function G(nu)
%
% Mathematical formulation:
%   min_x ||y - Hx||^2 + lambda * ||Dx||^2
%   X_hat(nu) = [ H*(nu) / (|H(nu)|^2 + lambda * |D(nu)|^2 + eps) ] * Y(nu)

    if nargin < 3 || isempty(lambda)
        lambda = 1e-4;
    end
    if nargin < 4 || isempty(reg_type)
        reg_type = 'gradient';
    end
    if nargin < 5 || isempty(center_psf)
        center_psf = true;
    end

    [rows, cols] = size(blurred);
    [p_rows, p_cols] = size(psf);

    % Pad PSF to full image grid
    psf_padded = zeros(rows, cols);
    psf_padded(1:p_rows, 1:p_cols) = psf;

    if center_psf
        shift_r = -floor(p_rows / 2);
        shift_c = -floor(p_cols / 2);
        psf_padded = circshift(psf_padded, [shift_r, shift_c]);
    end
    H = fft2(psf_padded);

    % Construct regularization kernel D
    switch lower(reg_type)
        case 'identity'
            D_kernel = [0 0 0; 0 1 0; 0 0 0];
        case 'gradient'
            D_kernel = [0 -1 0; -1 4 -1; 0 -1 0];
        case 'laplacian'
            D_kernel = [-1 -1 -1; -1 8 -1; -1 -1 -1];
        otherwise
            warning('Unknown reg_type "%s". Using "gradient".', reg_type);
            D_kernel = [0 -1 0; -1 4 -1; 0 -1 0];
    end

    [d_rows, d_cols] = size(D_kernel);
    D_padded = zeros(rows, cols);
    D_padded(1:d_rows, 1:d_cols) = D_kernel;

    if center_psf
        shift_dr = -floor(d_rows / 2);
        shift_dc = -floor(d_cols / 2);
        D_padded = circshift(D_padded, [shift_dr, shift_dc]);
    end
    D = fft2(D_padded);

    % Fourier transform of observed image
    Y = fft2(blurred);

    % Wiener-Hunt frequency filter
    denom = abs(H).^2 + lambda * abs(D).^2 + 1e-12;
    transfer_fn = conj(H) ./ denom;

    % Reconstruction
    X_hat = transfer_fn .* Y;
    restored = real(ifft2(X_hat));
end
