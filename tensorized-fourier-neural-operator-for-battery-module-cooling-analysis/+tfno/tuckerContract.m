function Y = tuckerContract(X, core, factors)
%TUCKERCONTRACT Apply a Tucker-factorized spectral convolution using precontracted spatial modes.
%  Y = TUCKERCONTRACT(X, core, factors) computes the output of a single
%  Fourier Neural Operator (FNO) spectral convolution layer where the
%  convolution kernel is represented in a Tucker factorization.
%
%  This implementation performs the contraction
%
%       Y =  U_out · ( core ×₃ U₁ ×₄ U₂ × ... ×_{d+2} U_d ) · U_inᵀ · X
%
%  without explicitly forming the full spectral weight tensor. Spatial
%  factors are first contracted into the core, then pagewise RC×RC
%  multiplications are used to efficiently apply the operator at each
%  spatial location and batch element.
%
%  Input and factor layouts:
%      X        – Input tensor of size [C, B, S₁, ..., S_d], where
%                 C  is the number of channels,
%                 B  is the batch size,
%                 S₁..S_d are the spatial sizes.
%
%      core     – Tucker core of size [RC, RC, R₁, ..., R_d], ordered as
%                    (rOut, rIn, r₁, ..., r_d).
%
%      factors  – Cell array {U_out, U_in, U₁, ..., U_d} where:
%                    U_out : [C  × RC]
%                    U_in  : [C  × RC]
%                    U_k   : [S_k × R_k] for k = 1..d
%
%  tuckerContract performs the following steps:
%      1) Compress input channels using  U_in.' * X.
%      2) Contract each spatial rank mode of the core with the
%         corresponding spatial factor U_k.
%      3) Perform pagewise RC×RC multiplications between the contracted
%         core and the compressed input at every (batch, spatial) index.
%      4) Expand rank outputs back to channel space using U_out.'
%
%  This function is optimized for minimal intermediate memory use and
%  avoids constructing large intermediate tensors.

% Copyright 2026 The MathWorks, Inc.

    arguments
        X
        core
        factors (1, :) cell
    end

    Uout = factors{1};
    UIn = factors{2};
    USpatial = factors(3:end);
    numSpatialDims = numel(USpatial);
    B = size(X, 2);
    RC = size(core, 1);

    % Spatial sizes [S1..Sd] from the first dim of Uk
    spatialSizes = cellfun(@(x) size(x, 1), USpatial);

    % Account for X being smaller than core/factors
    spatialSizes = min(spatialSizes, size(X, 3:numSpatialDims+2));

    % 1) Compress channels: X_I = UIn' * X
    X = pagemtimes(UIn.', X); % [RC, B, S1..Sd]

    % 2) Pre-contract spatial ranks into the core (once per call)
    %    For each spatial mode k, left-multiply core's rank mode r_k by Uk (size [Sk x Rk]).
    %    core: [RC, RC, R1, ..., Rd] → after all k: [RC, RC, S1, ..., Sd]
    for k = 1:numSpatialDims
        mode = 2 + k;
        % Account for X being smaller than the factor or core
        factor = USpatial{k}(1:spatialSizes(k), :);
        core = tfno.contractFactor(core, factor, mode);
    end

    % 3) Pagewise rank mixing (vectorized over batch and all spatial locations)
    %    Form pages so that each page holds one (b, s1..sd) location:
    %      core pages:  [RC, RC, 1, S1, ..., Sd]  (same per batch; broadcast on batch)
    %      X    pages:  [RC,  1, B, S1, ..., Sd]
    %    Compute Z_pages = core * X for all pages:
    %      Z_pages: [RC, 1, B, S1, ..., Sd] → squeeze/permute → Z: [B, S1, ..., Sd, RC]
    X = reshape(X, [RC, 1, B, spatialSizes]); % [RC, 1, B, S1..Sd]
    core = reshape(core, [RC, RC, 1, spatialSizes]); % [RC, RC, 1, S1..Sd]

    Z = pagemtimes(core, X); % [RC, 1, B, S1..Sd]
    Z = permute(squeeze(Z), [2:ndims(Z)-1, 1]); % [B, S1..Sd, RC]

    % 4) Expand to channels: Y_b = Z * Uout'  -> [B, S1..Sd, C]
    Y = reshape(reshape(Z, [], RC) * Uout.', [B, spatialSizes, size(Uout,1)]);

    % Set output as [S1..Sd, C, B]
    Y = permute(Y, [2:(numSpatialDims+2), 1]); % [S1..Sd, C, B]
end