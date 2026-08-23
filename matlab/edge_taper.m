function output = edge_taper(input_img, psf, varargin)
% EDGE_TAPER Backward compatibility wrapper for apply_edge_taper.
    output = apply_edge_taper(input_img, psf, varargin{:});
end
