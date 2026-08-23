function [restored, transfer_fn] = deconvolve(blurred, psf, lambda, varargin)
% DECONVOLVE Wrapper function for deconvolve_wiener.
%   Provides full backward compatibility with original function signature.
%
% Syntax:
%   [restored, transfer_fn] = deconvolve(blurred, psf, lambda, [reg_type], [center_psf])

    if nargin < 3
        lambda = 1e-4;
    end
    [restored, transfer_fn] = deconvolve_wiener(blurred, psf, lambda, varargin{:});
end
