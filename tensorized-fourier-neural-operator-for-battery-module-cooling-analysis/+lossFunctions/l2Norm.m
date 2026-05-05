function L2 = l2Norm(X, params)
%L2NORM Compute L2 norm on a grid.
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
%   Input X must be a numeric array of size [S1, S2, ..., SD, C, B]
%   where S1...SD are spatial dimensions, C is number of channels,
%   and B is batch size.
%
%   Example:
%     B=2; C=1; S1=64; S2=64;
%     X = randn(S1,S2,C,B);
%     L2 = l2Norm(X);

% Copyright 2026 The MathWorks, Inc.

    arguments
        X dlarray {mustBeNumeric}
        params.Reduction (1,1) string {mustBeMember(params.Reduction, {'mean', 'sum', 'none'})} = "mean"
        params.SquareRoot (1,1) logical = false
        params.Normalize (1,1) logical = false
    end

    sz = size(X);
    B = sz(end);

    % Reshape to [prod(S*C), B]
    X = reshape(X, [], B);

    L2 = sum(abs(X.^2), 1); % [1, B]

    if params.SquareRoot
        L2 = sqrt(L2);
    end

    if params.Reduction == "mean"
        L2 = mean(L2);
    elseif params.Reduction == "sum"
        L2 = sum(L2);
    end

    if params.Normalize
        L2 = L2 / prod(sz(1:end-1));
    end
end
