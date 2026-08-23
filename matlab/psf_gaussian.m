function kernel = psf_gaussian(sigma, size_kernel)
% PSF_GAUSSIAN Generates a 2D normalized Gaussian Point Spread Function.
%
% Syntax:
%   kernel = psf_gaussian(sigma)
%   kernel = psf_gaussian(sigma, size_kernel)

    if nargin < 1 || isempty(sigma)
        sigma = 1.0;
    end
    if nargin < 2 || isempty(size_kernel)
        size_kernel = ceil(6 * sigma);
        if mod(size_kernel, 2) == 0
            size_kernel = size_kernel + 1;
        end
    end

    half = floor(size_kernel / 2);
    [x, y] = meshgrid(-half:half, -half:half);

    kernel = exp(-(x.^2 + y.^2) / (2 * sigma^2));
    kernel = kernel / sum(kernel(:));
end
