function [ST_maps, eigenVal] = nullspace_vectors_G_matrix_2D(kCal, N1, N2, L, G, patchSize, ...
    varargin)

% Function that calculates the nullspace vectors for each G(x) matrix. These
% vectors correspond to spatiotemporal maps at the x location.
%
% Input parameters:
%   --kCal:         N1_cal x N2_cal x Nc block of calibration data, where
%                   N1_cal and N2_cal are the dimensions of a rectangular
%                   block of Nyquist-sampled k-space, and Nc is the number of
%                   channels in the array.
%
%   --N1, N2:       The desired dimensions of the output spatiotemporal maps.
%
%   --L:            Number of desired spatiotemporal maps to be calculated.
%
%   --G:            N1_g x N2_g x Nc x Nc array where G(i,j,:,:) 
%                   corresponds to the G matrix at the (i,j) spatial location.
%
%   --patchSize:    Number of elements in the kernel used to calculate the
%                   nullspace vectors of the C matrix.
%
%   --PowerIteration_G_nullspace_vectors: Binary variable. 0 = nullspace
%                   vectors of the G matrices are calculated using SVD.
%                   1 = nullspace vectors of the G matrices are calculated
%                   using the Power Iteration approach. Default: 1.
%
%   --M:            Number of iterations used in the Power Iteration approach
%                   to calculate the nullspace vectors of the G matrices.
%                   Default: 30.
%
%   --FFT_interpolation: Binary variable. 0 = no interpolation is used,
%                   1 = FFT-based interpolation is used. Default: 1.
%
%   --gauss_win_param: Parameter for the Gaussian apodizing window used to
%                   generate the low-resolution image in the FFT-based
%                   interpolation approach. This is the reciprocal of the
%                   standard deviation of the Gaussian window. Default: 100.
%
%   --verbose:      Binary variable. 1 = information about the convergence
%                   of Power Iteration is displayed. Default: 1.
%
% Output parameters:
%   --ST_maps:    N1 x N2 x Nc x L stack corresponding to the spatiotemporal
%                   maps for each channel present in the calibration data.
%
%   --eigenVal:     N1 x N2 x Nc array containing the eigenvalues of G(x)
%                   for each spatial location (normalized). Can be used for
%                   creating a mask describing the image support (e.g.,
%                   mask = (eigenVal(:,:,end) < 0.08);). If
%                   PowerIteration_G_nullspace_vectors == 1, only the
%                   smallest eigenvalue is returned (dimensions: N1 x N2).
%                   If FFT_interpolation == 1, approximations of eigenvalues
%                   are returned.

p = inputParser;

p.addRequired('kCal', @(x) isnumeric(x) && ndims(x) == 3);
p.addRequired('N1', @(x) isnumeric(x) && isscalar(x));
p.addRequired('N2', @(x) isnumeric(x) && isscalar(x));
p.addRequired('G', @(x) isnumeric(x) && ndims(x) == 4);
p.addRequired('patchSize', @(x) isnumeric(x) && isscalar(x));
p.addRequired('L', @(x) isnumeric(x) && isscalar(x));
p.addParameter('PowerIteration_G_nullspace_vectors', 1, @(x) isnumeric(x) && isscalar(x));
p.addParameter('M', 30, @(x) isnumeric(x) && isscalar(x));
p.addParameter('FFT_interpolation', 1, @(x) isnumeric(x) && isscalar(x));
p.addParameter('gauss_win_param', 100, @(x) isnumeric(x) && isscalar(x));
p.addParameter('verbose', 1, @(x) isnumeric(x) && isscalar(x));

if isempty(varargin)
    parse(p, kCal, N1, N2, G, patchSize, L);
else
    parse(p, kCal, N1, N2, G, patchSize, L, varargin{:});
end

N1_g = size(p.Results.G, 1);
N2_g = size(p.Results.G, 2);
Nc = size(p.Results.G, 3);
L = p.Results.L;
M = p.Results.M;

G = reshape(permute(p.Results.G, [3 4 1 2]), [Nc Nc (N1_g * N2_g)]);

if p.Results.PowerIteration_G_nullspace_vectors == 0
    [~, eigenVal, Vpage] = pagesvd(G, 'econ', 'vector');
    eigenVal = reshape(permute(eigenVal, [3 1 2]), [N1_g, N2_g, Nc]);
    ST_maps = reshape(permute(Vpage(:,Nc-L+1:Nc,:), [3 1 2]), [N1_g N2_g Nc L]);
    clear G
    eigenVal = eigenVal / p.Results.patchSize;
else
    G = G / p.Results.patchSize;
    G_null = repmat(eye(Nc), [1 1 (N1_g * N2_g)]);
    G_null = G_null - G;
    clear G

    S = randn(Nc,L) + 1i*randn(Nc,L);
    S = repmat(S, [1 1 N1_g*N2_g]);

    for m = 1:M
        S = pagemtimes(G_null, S);
        S(:,1,:) = S(:,1,:)./pagenorm(S(:,1,:));
        for l = 2:L
            for k = 1:l-1
                Rl = pagemtimes(S(:,k,:), 'ctranspose', S(:,l,:), 'none');
                S(:,l,:) = S(:,l,:) - Rl.*S(:,k,:);
            end
            S(:,l,:) = S(:,l,:)./pagenorm(S(:,l,:));
        end
    end
    E = pagemtimes(S, 'ctranspose', pagemtimes(G_null, S), 'none')./pagenorm(S);
    S = permute(S, [3 1 2]);
    ST_maps = reshape(S, [N1_g N2_g Nc, L]);
    eigenVal_aux = reshape(permute(E, [3 1 2]), [N1_g, N2_g, L, L]);
    eigenVal_aux = 1 - eigenVal_aux;

    eigenVal = zeros(N1_g, N2_g, L);

    for l = 1:L
        eigenVal(:,:,l) = squeeze(eigenVal_aux(:,:,l,l));
    end
    clear E S G_null eigenVal_aux

end

% ==== FFT-based interpolation ====
if p.Results.FFT_interpolation == 1
    [N1_cal, N2_cal, ~] = size(kCal);
    w_sm = (0.54 - 0.46 * cos(2 * pi * ((0:(N1_g - 1)) / (N1_g - 1))))';
    w_sm2 = (0.54 - 0.46 * cos(2 * pi * ((0:(N2_g - 1)) / (N2_g - 1))))';
    w_sm = w_sm * w_sm2';
    w_sm = repmat(w_sm, [1 1 Nc]);

    T = fftshift(fft2(ifftshift(eigenVal)));
    T = T .* w_sm(:, :, end);
    eigenVal = abs(fftshift(fftshift(ifft2(T, p.Results.N1, p.Results.N2), 1), 2));
    eigenVal = eigenVal / max(eigenVal(:));

    % FFT-based interpolation of spatiotemporal maps
    win1 = gausswin(N1_g, p.Results.gauss_win_param);
    win2 = gausswin(N2_g, p.Results.gauss_win_param)';
    apod2D = win1 * win2;

    imLowRes_cal = zeros(N1_g, N2_g, Nc, 'like', ST_maps);
    cx = ceil(N1_g / 2) + utils.even_pisco(N1_g / 2);
    cy = ceil(N2_g / 2) + utils.even_pisco(N2_g / 2);
    hx = floor(N1_cal / 2);
    hy = floor(N2_cal / 2);
    rowIdx = cx + (-hx: hx - utils.even_pisco(N1_cal / 2));
    colIdx = cy + (-hy: hy - utils.even_pisco(N2_cal / 2));
    imLowRes_cal(rowIdx, colIdx, :) = p.Results.kCal;

    tmp = imLowRes_cal .* apod2D;
    tmp = ifftshift(tmp);
    imLowRes_cal = fftshift(ifft2(tmp));

    num = sum(conj(ST_maps) .* imLowRes_cal, 3);
    den = sum(abs(ST_maps) .^ 2, 3);
    cim = num ./ den;
    phase_cim = exp(1i * angle(cim));
    ST_maps = ST_maps .* phase_cim;

    S = fftshift(fft2(ifftshift(ST_maps)));
    S = S .* w_sm;
    ST_maps = fftshift(fftshift(ifft2(S, p.Results.N1, p.Results.N2), 1), 2);
end

end