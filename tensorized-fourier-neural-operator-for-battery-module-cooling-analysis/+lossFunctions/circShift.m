function Y = circShift(X, k, d)
%CIRCSHIFT Circular shift along dimension d by k positions.
%   Y = CIRCSHIFT(X, k, d) shifts the elements of X by k positions along
%   dimension d with wrapping. Positive k shifts toward higher indices
%   (shift right), negative k shifts toward lower indices (shift left).
%
%   Uses explicit subscript indexing rather than MATLAB's built-in
%   circshift to guarantee dlarray compatibility with dlgradient and
%   dlaccelerate.

% Copyright 2026 The MathWorks, Inc.

    n = size(X, d);
    k = mod(k, n);
    if k == 0
        Y = X;
        return
    end
    idx = repmat({':'}, 1, ndims(X));
    idx{d} = [n-k+1:n, 1:n-k];
    Y = X(idx{:});
end
