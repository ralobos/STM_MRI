function [ST_maps, eigenValues] = STM_computation(kCal, dim_sens, L, varargin)

    % Input parameters:
    %   --kCal:                            2D case: N1_cal x N2_cal x Nc block of calibration data, where 
    %                                               N1_cal and N2_cal are the dimensions of a rectangular 
    %                                               block of Nyquist-sampled k-space, and Nc is the number of 
    %                                               channels in the array.
    %
    %   --dim_sens:                        2D case: 1x2 array with the desired dimensions of the output 
    %                                               spatiotemporal maps.
    %
    %   --L:                               Number of desired spatiotemporal maps to be calculated.
    %
    %   --tau:                             2D case: Parameter (in Nyquist units) that determines the size of 
    %                                               the k-space kernel. For a rectangular kernel, the size is 
    %                                               (2*tau+1) x (2*tau+1). For an ellipsoidal kernel, it is 
    %                                               the radius of the associated neighborhood. Default: 3.
    %
    %   --threshold:                       Specifies how small a singular value needs to be (relative 
    %                                      to the maximum singular value) before its associated 
    %                                      singular vector is considered to be in the nullspace of 
    %                                      the C-matrix. Default: 0.05.
    %
    %   --kernel_shape:                    Binary variable. 0 = rectangular kernel, 1 = ellipsoidal 
    %                                      kernel. Default: 1.
    %
    %   --FFT_nullspace_C_calculation:     Binary variable. 0 = nullspace vectors 
    %                                      of C are calculated from C'*C by calculating C first. 
    %                                      1 = nullspace vectors of C are calculated from C'*C 
    %                                      directly using an FFT-based approach. Default: 1.
    %
    %   --OrthogonalIteration_G_nullspace_vectors: Binary variable. 0 = nullspace 
    %                                         vectors of the G matrices are calculated using SVD. 
    %                                         1 = nullspace vectors of the G matrices are calculated 
    %                                         using a Orthogonal Iteration approach. Default: 1.
    %
    %   --M:                               Number of iterations used in the Orthogonal Iteration approach 
    %                                      to calculate the nullspace vectors of the G matrices. 
    %                                      Default: 30.
    %
    %   --FFT_interpolation:               Binary variable. 0 = no interpolation. 1 = 
    %                                      FFT-based interpolation is used. Default: 1.
    %
    %   --interp_zp:                       Amount of zero-padding to create the low-resolution grid 
    %                                      if FFT-interpolation is used. 
    %                                      2D case: The grid has dimensions 
    %                                               (N1_cal + interp_zp) x (N2_cal + interp_zp) x Nc. 
    %                                      Default: 24.
    %
    %   --gauss_win_param:                 Parameter for the Gaussian apodizing window used to 
    %                                      generate the low-resolution image in the FFT-based 
    %                                      interpolation approach. This is the reciprocal of the 
    %                                      standard deviation of the Gaussian window. Default: 100.
    %
    %   --sketched_SVD:                    Binary variable. 1 = sketched SVD is used to calculate 
    %                                      a basis for the nullspace of the C matrix. Default: 1.
    %
    %   --sketch_dim:                      Dimension of the sketch matrix used to calculate a basis 
    %                                      for the nullspace of the C matrix using a sketched SVD. 
    %                                      Only used if sketched_SVD is enabled. Default: 500.
    %
    %   --visualize_C_matrix_sv:           Binary variable. 1 = Singular values of the C matrix are displayed.
    %                                      Default: 0. 
    %                                      Note: If sketched_SVD = 1 and if the curve of the singular values flattens out,
    %                                      it suggests that the sketch dimension is appropriate for the data.
    %
    %   --verbose:                         Binary variable. 1 = display PISCO information, including 
    %                                      which techniques are employed and computation times for 
    %                                      each step. Default: 1.
    %
    % Output parameters:
    %   --ST_maps:                       2D case: dim_sens(1) x dim_sens(2) x Nc x L stack corresponding to the 
    %                                               spatiotemporal maps for each channel present in the 
    %                                               calibration data.
    %
    %   --eigenValues:                     2D case: dim_sens(1) x dim_sens(2) x Nc array containing the 
    %                                               eigenvalues of G(x) for each spatial location (normalized 
    %                                               by the kernel size). Can be used for creating a mask
    %                                               describing the image support (e.g., mask = 
    %                                               (eigenValues(:,:,end) < 0.08);). If
    %                                               PowerIteration_G_nullspace_vectors == 1, only the smallest
    %                                               eigenvalues are returned (dimensions: dim_sens(1) x
    %                                               dim_sens(2) x L). If FFT_interpolation == 1, approximations
    %                                               of eigenvalues are returned.

    % V1.0: Rodrigo A. Lobos (rlobos@umich.edu)
    % December, 2025.

    % Set default values for optional parameters
    p = inputParser;

    addRequired(p, 'kCal', @(x) isnumeric(x) && (ndims(x) == 3 || ndims(x) == 4));
    addRequired(p, 'dim_sens', @(x) isnumeric(x) && isvector(x) && (length(x) == 2 || length(x) == 3));
    addRequired(p, 'L', @(x) isnumeric(x) && isscalar(x));

    addParameter(p, 'tau', 3, @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'threshold', 0.05, @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'kernel_shape', 1, @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'FFT_nullspace_C_calculation', 1, @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'OrthogonalIteration_G_nullspace_vectors', 1, @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'M', 30, @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'FFT_interpolation', 1, @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'interp_zp', 24, @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'gauss_win_param', 100, @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'sketched_SVD', 1, @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'sketch_dim', 500, @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'visualize_C_matrix_sv', 0, @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'verbose', 1, @(x) isnumeric(x) && isscalar(x));

    if isempty(varargin)
        parse(p, kCal, dim_sens, L);
    else
        parse(p, kCal, dim_sens, L, varargin{:});
    end

    if p.Results.verbose == 1
        if p.Results.kernel_shape == 0
            kernel_shape_q = 'Rectangular';
        else
            kernel_shape_q = 'Ellipsoidal';
        end

        if p.Results.FFT_nullspace_C_calculation == 0
            FFT_nullspace_C_calculation_q = 'No';
        else
            FFT_nullspace_C_calculation_q = 'Yes';
        end

        if p.Results.FFT_interpolation == 0
            FFT_interpolation_q = 'No';
        else
            FFT_interpolation_q = 'Yes';
        end

        if p.Results.OrthogonalIteration_G_nullspace_vectors == 0
            OrthogonalIteration_nullspace_vectors_q = 'No';
        else
            OrthogonalIteration_nullspace_vectors_q = 'Yes';
        end

        if p.Results.sketched_SVD == 0
            sketched_SVD_q = 'No';
        else
            sketched_SVD_q = 'Yes';
        end

        disp('Selected PISCO techniques:')
        disp('=======================')
        disp(['Kernel shape : ' kernel_shape_q])
        disp(['FFT-based calculation of nullspace vectors of C : ' FFT_nullspace_C_calculation_q])
        disp(['Sketched SVD for nullspace vectors of C : ' sketched_SVD_q])
        disp(['FFT-based interpolation : ' FFT_interpolation_q])
        disp(['OrthogonalIteration-based nullspace estimation for G matrices : ' OrthogonalIteration_nullspace_vectors_q])
        disp('=======================')
    end

    t_null = tic;

    % ==== Nullspace-based algorithm Steps (1) and (2)  ====
    % Calculation of nullspace vectors of C
    t_null_vecs = tic;

    opts_nullspace_C_matrix = struct( ...
        'tau',                        p.Results.tau, ...
        'threshold',                  p.Results.threshold, ...
        'kernel_shape',               p.Results.kernel_shape, ...
        'FFT_nullspace_C_calculation', p.Results.FFT_nullspace_C_calculation, ...
        'sketched_SVD',               p.Results.sketched_SVD, ...
        'sketch_dim',                 p.Results.sketch_dim, ...
        'visualize_C_matrix_sv',      p.Results.visualize_C_matrix_sv ...
    );

    fn = fieldnames(opts_nullspace_C_matrix);
    fv = struct2cell(opts_nullspace_C_matrix);
    nv = [fn.'; fv.'];
    nv = nv(:).';

    U = utils.nullspace_vectors_C_matrix_2D(kCal, nv{:});

    t_null_vecs = toc(t_null_vecs);

    if p.Results.verbose == 1
        if p.Results.FFT_nullspace_C_calculation == 0
            aux_word = 'Calculating C first';
        else
            aux_word = 'FFT-based direct calculation of ChC';
        end

        if p.Results.sketched_SVD == 0
            aux_word = [aux_word ', using regular SVD'];
        else
            aux_word = [aux_word ', using sketched SVD'];
        end

        disp('=======================')
        disp('PISCO computation times (secs):')
        disp('=======================')
        disp(['Time nullspace vectors of C (' aux_word ') : ' num2str(t_null_vecs)])
        disp('=======================')
    end

    % ==== Nullspace-based algorithm Step (3)  ====
    % Direct computation of G matrices
    t_G_matrices = tic;

    opts_G_matrices = struct( ...
        'kernel_shape',  p.Results.kernel_shape, ...
        'FFT_interpolation', p.Results.FFT_interpolation, ...
        'interp_zp',     p.Results.interp_zp, ...
        'sketched_SVD',  p.Results.sketched_SVD ...
    );

    fn = fieldnames(opts_G_matrices);
    fv = struct2cell(opts_G_matrices);
    nv = [fn.'; fv.'];
    nv = nv(:).';

    G = utils.G_matrices_2D( ...
        kCal, dim_sens(1), dim_sens(2), p.Results.tau, U, nv{:} ...
    );

    t_G_matrices = toc(t_G_matrices);

    Nc = size(kCal, 3);
    patchSize = size(U, 1) / Nc;
    clear U


    if p.Results.verbose == 1
        disp(['Time G matrices (direct calculation): ' num2str(t_G_matrices)])
        disp('=======================')
    end

    % ==== Nullspace-based algorithm Step (4)  ====
    % Calculation of nullspace vectors of the G matrices
    t_null_G = tic;

    opts_G_nullspace_vectors = struct( ...
        'OrthogonalIteration_G_nullspace_vectors', p.Results.OrthogonalIteration_G_nullspace_vectors, ...
        'M',                                  p.Results.M, ...
        'FFT_interpolation',                  p.Results.FFT_interpolation, ...
        'gauss_win_param',                    p.Results.gauss_win_param, ...
        'verbose',                            p.Results.verbose ...
    );

    fn = fieldnames(opts_G_nullspace_vectors);
    fv = struct2cell(opts_G_nullspace_vectors);
    nv = [fn.'; fv.'];
    nv = nv(:).';

    [ST_maps, eigenValues] = utils.nullspace_vectors_G_matrix_2D( ...
        kCal, dim_sens(1), dim_sens(2), L, G, patchSize, nv{:} ...
    );

    t_null_G = toc(t_null_G);

    clear G

    if p.Results.verbose == 1
        if p.Results.OrthogonalIteration_G_nullspace_vectors == 0
            aux_word = 'Using SVD';
        else
            aux_word = 'Using Orthogonal Iteration';
        end

        disp(['Time nullspace vector G matrices (' aux_word ') : ' num2str(t_null_G)])
        disp('=======================')
    end

    % ==== Nullspace-based algorithm Step (5)  ====
    % Normalization

    % Phase-reference all coils to the first coil 
    phase_ref = exp(-1i * angle(ST_maps(:, :, 1)));
    ST_maps = ST_maps .* phase_ref;

    % Normalize sensitivities to unit L2 norm across coils at each pixel
    den = sqrt(sum(abs(ST_maps).^2, 3));
    den(den == 0) = 1;
    ST_maps = ST_maps ./ den;

    if p.Results.verbose == 1
        disp(['Total time: ' num2str(toc(t_null))])
        disp('=======================')
    end

end