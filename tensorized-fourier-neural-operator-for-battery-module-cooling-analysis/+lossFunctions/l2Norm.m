function L2 = l2Norm(X, params)
%L2NORM - Compute L2 norm on a grid.
%   L2 = L2NORM(X) computes the L2 norm of the input array X
%   with default parameters.
%
%   L2 = L2NORM(X, Name=Value) specifies additional options using
%   one or more name-value arguments:
%
%     Reduction     - Method for reducing the norm across batch.
%                     Options are 'mean', 'sum', or 'none'.
%                     The default value is 'mean'.
%
%     SquareRoot    - If false, returns the squared L2 norm.
%                     If true, returns the L2 norm. The default
%                     value is false.
%
%     Normalize     - If true, divides output by C*prod(S1, S2, ...).
%                     The default value is false.
%
%   Input X must be a numeric array of size [B, C, S1, S2, ..., SD]
%   where B is batch size, C is number of channels, and S1...SD are
%   spatial dimensions.
%
%   Example:
%     B=2; C=1; S1=64; S2=64;
%     X = randn(B,C,S1,S2);
%     L2 = l2Norm(X);

% Copyright 2026 The MathWorks, Inc.

    arguments
        X dlarray {mustBeNumeric}
        params.Reduction (1,1) string {mustBeMember(params.Reduction, {'mean', 'sum', 'none'})} = "mean"
        params.SquareRoot (1,1) logical = false
        params.Normalize (1,1) logical = false
    end

    sz = size(X);

    % Convert to BxCS
    X = reshape(X, sz(1), []);

    L2 = sum(abs(X.^2), 2); % Bx1, abs() needed for complex values

    if params.SquareRoot
        L2 = sqrt(L2);
    end

    if params.Reduction == "mean"
        L2 = mean(L2);
    elseif params.Reduction == "sum"
        L2 = sum(L2);
    end

    if params.Normalize
        L2 = L2/(prod(sz(2:end)));
    end

end