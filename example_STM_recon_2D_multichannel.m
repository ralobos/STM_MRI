%% SPATIOTEMPORAL MAPS RECONSTRUCTION EXAMPLE
%
% This script performs reconstruction of 2D multichannel retrospectively
% undersampled dynamic MRI data using spatiotemporal maps (STM) computed
% from calibration data.
%
% A Tikhonov-regularized STM reconstruction is also performed.
%
% The reconstruction is performed using a conjugate-gradient (CG)
% approach.
%
% The script uses the PISCO techniques for efficient computation of
% spatiotemporal maps.
%
% The spatiotemporal maps reconstruction framework is described in:
%
% [1] R. A. Lobos, X. Wang, R. T. L. Fung, Y. He, D. Frey, D. Gupta,
%     Z. Liu, J. A. Fessler, D. C. Noll. Spatiotemporal Maps for Dynamic MRI
%     Reconstruction, 2025, arXiv:2507.14429.
%     (https://arxiv.org/abs/2507.14429)
%
% Author:  Rodrigo A. Lobos (rlobos@umich.edu)
% Date:    December 2025


clear all;
close all;
clc;

%% Loading data

load('./data/2D_mc_cardiac.mat');

% idata_gt    - ground truth data
% idata_gt_sc - ground truth single-coil data obtained using a SENSE
%               coil-combination
% kmask       - sampling mask
% sense_maps  - sensitivity maps

%% Data dimensions

[N1, N2, Nc, Nt] = size(idata_gt);  % N1 x N2 : image dimensions
                                    % Nc      : number of coils
                                    % Nt      : number of time frames

%% sum-of-squares coil combination

idata_gt_sos = squeeze(sqrt(sum(abs(idata_gt).^2,3)));

%% Visualization of all the frames

figure;
imagesc(utils.mdisp(abs(idata_gt_sos))); 
colormap gray; 
axis tight;
axis image;
axis off;
title('Ground truth data - all frames');

%% Visualization of k-space mask (all frames)

figure;
imagesc(utils.mdisp(squeeze(kmask(:, :, 1, :)))); 
colormap gray; 
axis tight;
axis image;
axis off;
title('k-space sampling mask - all frames');

%% Ground-truth data in k-space

kdata_gt = utils.ft2(idata_gt);

%% Retrospectively undersampled k-space data

kdata_under = kdata_gt .* kmask;

%% sum-of-squares coil combination of undersampled data

idata_under_sos = squeeze(sqrt(sum(abs(utils.ift2(kdata_under)).^2,3)));
figure;
imagesc(utils.mdisp(abs(idata_under_sos))); 
colormap gray;
axis tight;
axis image;
axis off;
title('Undersampled data - all frames');

%% Calibration data parameters

cal_length = 8;  % PE lines used for calibration

center_y = ceil(N2/2) + utils.even_pisco(N2);

cal_index_x = 1:N1;
cal_index_y = center_y + [-floor(cal_length/2):floor(cal_length/2) - utils.even_pisco(cal_length/2)];

%% Low-resolution SENSE reconstruction for STM computation

kdata_low_res_mc = zeros(N1, N2, Nc, Nt);

kdata_low_res_mc(:, cal_index_y, :, :) = kdata_under(:, cal_index_y, :, :);

idata_low_res_mc = utils.ift2(kdata_low_res_mc);

sense_recon_lr = zeros(N1, N2, Nt);

for t = 1:Nt
    Qlr = sum(conj(sense_maps).*idata_low_res_mc(:,:,:,t), 3)./sum(abs(sense_maps).^2, 3);
    Qlr(isnan(Qlr))=0;  
    sense_recon_lr(:,:,t) = Qlr;
end

kdata_low_res = utils.ft2(sense_recon_lr);

% Visualization of low-resolution SENSE reconstruction

figure;
imagesc(utils.mdisp(abs(sense_recon_lr))); 
colormap gray;
axis tight;
axis image;
axis off;
title('Low-resolution SENSE reconstruction (sum-of-squares) - all frames');

%% STM computation

% Calibration data
    
N1_cal = N1;
N2_cal = cal_length; % Number of PE lines used for calibration

kCal = kdata_low_res(cal_index_x, cal_index_y, :);

t_total = tic;


%% STM computation parameters

dim_sens = [N1, N2];                  % Desired dimensions for the computed spatiotemporal maps.

L = 4;                                % Number of temporal basis functions to be computed.

tau      = 3;                         % Kernel radius. Default: 3

threshold = 0.08;                     % Threshold for C-matrix singular values. Default: 0.05
                                      % Note: In this example we don't use the default value.

M = 20;                               % Number of iterations for Power Iteration. Default: 30
                                      % Note: In this example we use a smaller value
                                      % to speed up the calculations.

interp_zp = 100;                       % Amount of zero-padding to create the low-resolution grid 
                                      % if FFT-interpolation is used. Default: 24

gauss_win_param = 100;                % Parameter for the Gaussian apodizing window used to 
                                      % generate the low-resolution image in the FFT-based 
                                      % interpolation approach. This is the reciprocal of the 
                                      % standard deviation of the Gaussian window. Default: 100

sketch_dim = 300;                     % Dimension of the sketch matrix used to calculate a
                                      % basis for the nullspace of the C matrix using a sketched SVD. 
                                      % Default: 500. Note: In this example we use a smaller value
                                      % to speed up the calculations.

visualize_C_matrix_sv = 1;            % Binary variable. 1 = Singular values of the C matrix are displayed.
                                      % Default: 0. 
                                      % Note: In this example we set it to 1 to visualize the singular values
                                      % of the C matrix. If sketched_SVD = 1 and if the curve of the singular values flattens out,
                                      % it suggests that the sketch dimension is appropriate for the data.
                                      
%% PISCO techniques

% The following techniques are used if the corresponding binary variable is equal to 1

kernel_shape = 1;                     % Binary variable. 1 = ellipsoidal shape is adopted for 
                                      % the calculation of kernels (instead of rectangular shape).
                                      % Default: 1

FFT_nullspace_C_calculation = 1;      % Binary variable. 1 = FFT-based calculation of nullspace 
                                      % vectors of C by calculating C'*C directly (instead of 
                                      % calculating C first). Default: 1

sketched_SVD = 1;                     % Binary variable. 1 = sketched SVD is used to calculate 
                                      % a basis for the nullspace of the C matrix (instead of 
                                      % calculating the nullspace vectors directly and then the 
                                      % basis). Default: 1

PowerIteration_G_nullspace_vectors = 1; % Binary variable. 1 = Power Iteration approach is 
                                        % used to find nullspace vectors of the G matrices 
                                        % (instead of using SVD). Default: 1

FFT_interpolation = 1;                % Binary variable. 1 = sensitivity maps are calculated on 
                                      % a small spatial grid and then interpolated to a grid with 
                                      % nominal dimensions using an FFT-approach. Default: 1

verbose = 1;                          % Binary variable. 1 = PISCO information is displayed. 
                                      % Default: 1

%% PISCO estimation

if isempty(which('STM_computation'))
    error(['The function STM_computation.m is not found in your MATLAB path. ' ...
           'Please ensure that all required files are available and added to the path.']);
end

t_stm = tic;

[ST_maps, eigenValues] = STM_computation( ...
    kCal, ...
    dim_sens, ...                          % Data and output size
    L, ...
    'tau', tau, ...
    'threshold', threshold, ...
    'kernel_shape', kernel_shape, ...            % Kernel and threshold parameters
    'FFT_nullspace_C_calculation', FFT_nullspace_C_calculation, ...             % FFT nullspace calculation flag
    'PowerIteration_G_nullspace_vectors', PowerIteration_G_nullspace_vectors, ...      % Power Iteration flag
    'M', M, ...
    'FFT_interpolation', FFT_interpolation, ...
    'interp_zp', interp_zp, ...
    'gauss_win_param', gauss_win_param, ... % Interpolation params
    'sketched_SVD', sketched_SVD, ...
    'sketch_dim', sketch_dim, ...
    'visualize_C_matrix_sv', visualize_C_matrix_sv, ... % SVD/sketching params
    'verbose', verbose ...                                  % Verbosity
);

disp(['Time for STM computation: ' num2str(toc(t_stm)) ' seconds']);
disp('=======================');

figure; 
utils.mdisp(abs(eigenValues));
axis tight;
axis image;
colorbar; 
colormap gray;
clim([0 1])
title('Eigenvalues of G matrices (normalized)');

%% STM reconstruction

N = N1*N2;

% Operators to apply spatiotemporal maps

Sv = @(x) utils.vect(sum(ST_maps.*repmat(reshape(x, [N1 N2 1 L]), [1 1 Nt 1]), 4)); % Operators to apply temporal maps
Sv_h = @(x) utils.vect(sum(conj(ST_maps).*repmat(reshape(x, [N1 N2 Nt 1]), [1 1 1 L]), 3));
Sv_h_Sv = @(x) Sv_h(Sv(x));

% Operators to apply sensitivity maps

Fv = @(x) utils.vect(repmat(reshape(x, [N1 N2 1 Nt]), [1 1 Nc 1]) .* repmat(sense_maps, [1 1 1 Nt]));
Fv_h = @(x) utils.vect(sum(conj(sense_maps).*reshape(x, [N1 N2 Nc Nt]), 3));

% Forward system operator

A = @(x) utils.vect(kmask.*utils.ft2(reshape(Fv(Sv(x)), [N1 N2 Nc Nt])));

% Adjoint system operator

Ah = @(x) Sv_h(Fv_h(utils.vect(utils.ift2(kmask.*reshape(x, [N1 N2 Nc Nt])))));

% Composition operator

AhA =  @(x) Ah(A(x));

%% CG-STM reconstruction

% This is a simple reconstruction with no regularization

disp('Starting CG-STM reconstruction...');
disp('=======================');

t_stm_recon = tic;

[z, ~] = pcg(AhA, Ah(kdata_under), 1e-6, 50); % Reconstructed STM spatial coefficients

sense_recon_stm = reshape(Sv(z), [N1 N2 Nt]);

disp(['Time for CG-STM reconstruction: ' num2str(toc(t_stm_recon)) ' seconds']);
disp('=======================');

% NRMSE CG-STM reconstruction

NRMSE_stm = norm(idata_gt_sc(:) - sense_recon_stm(:))/norm(idata_gt_sc(:));

disp(['NRMSE CG-STM reconstruction: ' num2str(NRMSE_stm)]);
disp('=======================');

% Visualization of CG-STM reconstruction

figure;
imagesc(utils.mdisp(abs(sense_recon_stm))); 
colormap gray;
axis tight;
axis image;
axis off;
title('CG-STM reconstruction - all frames');

%% CG-STM + Tikhonov reconstruction

% Reconstruction with a simple Tikhonov regularization

lambda_tik = 0.001; % Tikhonov regularization parameter

AtikA = @(x) AhA(x) + lambda_tik*x;

disp('Starting CG-STM + Tikhonov reconstruction...');
disp('=======================');

t_stm_recon_tik = tic;

[z_tik, ~] = pcg(AtikA, Ah(kdata_under), 1e-6, 50); % Reconstructed STM spatial coefficients

sense_recon_stm_tik = reshape(Sv(z_tik), [N1 N2 Nt]);

disp(['Time for CG-STM + Tikhonov reconstruction: ' num2str(toc(t_stm_recon_tik)) ' seconds']);
disp('=======================');

% NRMSE CG-STM + Tikhonov reconstruction

NRMSE_stm_tik = norm(idata_gt_sc(:) - sense_recon_stm_tik(:))/norm(idata_gt_sc(:));

disp(['NRMSE CG-STM + Tikhonov reconstruction: ' num2str(NRMSE_stm_tik)]);
disp('=======================');

% Visualization of CG-STM + Tikhonov reconstruction

figure;
imagesc(utils.mdisp(abs(sense_recon_stm_tik))); 
colormap gray;
axis tight;
axis image;
axis off;
title('CG-STM + Tikhonov reconstruction - all frames');
