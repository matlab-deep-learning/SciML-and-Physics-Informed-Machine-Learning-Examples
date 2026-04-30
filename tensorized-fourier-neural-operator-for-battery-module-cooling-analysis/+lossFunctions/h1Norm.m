function H1 = h1Norm(X, params)
%H1NORM Compute H1 norm on a grid.
%   H1 = H1NORM(X) computes the H1 norm of the input array X
%   with default parameters.
%
%   H1 = H1NORM(X, Name=Value) specifies additional options using
%   one or more name-value arguments:
%
%     Spacings      - 1xD vector of grid spacings [Δ1, Δ2, ..., ΔD].
%                     The default value is ones(1,D).
%
%     IncludeL2     - If true, computes full H1 norm (L2 + gradient).
%                     If false, computes seminorm only (gradient).
%                     The default value is true.
%
%     Reduction     - Method for reducing the norm across batch.
%                     Options are 'mean', 'sum', or 'none'.
%                     The default value is 'mean'.
%
%     Periodic      - 1xD logical array indicating which spatial
%                     dimensions are periodic. The default value
%                     is true for all dimensions.
%
%     SquareRoot    - If false, returns the squared H1 norm.
%                     If true, returns the H1 norm. The default
%                     value is false.
%
%     Normalize     - If true, divides output by C*prod(S1, S2, ...).
%                     The default value is false.
%
%   The H1 norm is defined as:
%     ||u||_{H^1} = (||u||_{L^2}^2 + ||∇u||_{L^2}^2)^{1/2}
%   where ||∇u||_{L^2}^2 = Σ_i ||∂u/∂x_i||_{L^2}^2.
%
%   Input X must be a numeric array of size [S1, S2, ..., SD, C, B]
%   where S1...SD are spatial dimensions, C is number of channels,
%   and B is batch size.
%
%   Gradients are estimated using central differences and one-sided
%   differences at boundaries (unless periodic boundary conditions).
%
%   Example:
%     B=2; C=1; S1=64; S2=64;
%     X = randn(S1,S2,C,B);
%     H1 = h1Norm(X);

% Copyright 2026 The MathWorks, Inc.

    arguments
        X dlarray {mustBeNumeric}
        params.Spacings (1,:) double = []
        params.IncludeL2 (1,1) logical = true
        params.Reduction (1,1) string {mustBeMember(params.Reduction, {'mean', 'sum', 'none'})} = "mean"
        params.Periodic (1,:) logical = true
        params.SquareRoot (1,1) logical = false
        params.Normalize (1,1) logical = false
    end

    sz = size(X);
    nd = ndims(X);
    if nd < 3
        error('Input must be at least [S1, C, B].');
    end
    B = sz(nd);
    C = sz(nd-1);
    D = nd - 2;
    spatialSizes = sz(1:D);

    if isempty(params.Spacings)
        params.Spacings = ones(1, D);
    else
        if numel(params.Spacings) ~= D
            error('Spacings must have length equal to the number of spatial dimensions (D).');
        end
    end

    if isscalar(params.Periodic)
        params.Periodic = repmat(params.Periodic, 1, D);
    elseif numel(params.Periodic) ~= D
        error('Periodic must be scalar or 1xD logical.');
    end

    % Initialize H1 as the L2 error,
    if params.IncludeL2
        H1 = lossFunctions.l2Norm(X, Reduction="none", SquareRoot=false, Normalize=false);
    else
        H1 = zeros(1, B, 'like', X);
    end

    % Reshape to [S1, ..., SD, C*B] so all batch/channel combinations are
    % handled independently along the trailing dimension.
    X = reshape(X, [spatialSizes, C*B]);

    for d = 1:D
        delta = params.Spacings(d);

        Xfwd = lossFunctions.circShift(X, -1, d);
        Xbwd = lossFunctions.circShift(X, 1, d);
        fd = (Xfwd - Xbwd) / (2 * delta);

        if ~params.Periodic(d)
            % Replace first/last elements with forward/reverse differences.

            if min(spatialSizes) < 4
                error("Non-periodic dimensions require at least 4 grid points for 3rd-order differences.");
            end
            fd = applyThirdOrderDifferenceAtBoundary(fd, X, d, delta);
        end

        fd = fd.^2;

        % Reshape back to [S1, ..., SD, C, B] and sum over all non-batch dims.
        fd = reshape(fd, sz);
        fd = sum(fd, 1:nd-1);
        fd = reshape(fd, 1, B);

        % Accumulate per-batch sum.
        H1 = H1 + fd;
    end

    if params.SquareRoot
        H1 = sqrt(H1);
    end

    if params.Normalize
        % Normalize by channels and number of spatial points
        H1 = H1 / (C * prod(spatialSizes));
    end

    if strcmp(params.Reduction, "mean")
        H1 = mean(H1, 'all');
    elseif strcmp(params.Reduction, "sum")
        H1 = sum(H1, 'all');
    end
end

function fd = applyThirdOrderDifferenceAtBoundary(fd, X, d, delta)

    % Get the indices of components for 3rd-order forward differences.
    idx1 = makeIndex(ndims(fd), d, 1);
    idx2 = makeIndex(ndims(fd), d, 2);
    idx3 = makeIndex(ndims(fd), d, 3);
    idx4 = makeIndex(ndims(fd), d, 4);

    fd(idx1{:}) = (-11*X(idx1{:}) + 18*X(idx2{:}) - 9*X(idx3{:}) + 2*X(idx4{:})) / (6 * delta);

    % Get the indices of components for 3rd-order backward differences.
    sz = size(fd, d);
    idx1 = makeIndex(ndims(fd), d, sz);
    idx2 = makeIndex(ndims(fd), d, sz-1);
    idx3 = makeIndex(ndims(fd), d, sz-2);
    idx4 = makeIndex(ndims(fd), d, sz-3);

    % Apply 3rd-order backward differences at right boundary
    fd(idx1{:}) = (-2*X(idx4{:}) + 9*X(idx3{:}) - 18*X(idx2{:}) + 11*X(idx1{:})) / (6 * delta);
end

function idx = makeIndex(ndims, toChange, val)
    idx = repmat({':'}, 1, ndims);
    idx{toChange} = val;
end
