function out = ift2(x)
% Centered 2D inverse Fourier transform
%
% Input:
%   x - 2D k-space data or multi-dimensional array
%
% Output:
%   out - Inverse Fourier transform with proper centering

out = fftshift(ifft2(ifftshift(x)));