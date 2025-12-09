function out = ft2(x)
% Centered 2D Fourier transform
%
% Input:
%   x - 2D image or multi-dimensional array
%
% Output:
%   out - Fourier transform with DC component centered
%

out = fftshift(fft2(ifftshift(x)));