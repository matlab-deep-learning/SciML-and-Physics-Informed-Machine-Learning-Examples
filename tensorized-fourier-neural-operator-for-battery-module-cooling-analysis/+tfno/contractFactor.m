function A = contractFactor(A, factor, mode)
%CONTRACTFACTOR - Contract a tensor along a specified mode with a factor matrix.
%  CONTRACTFACTOR performs mode-n multiplication of a tensor A with a factor matrix.
%  The operation contracts the tensor along the specified mode by multiplying with
%  the transpose of the factor matrix.
%
%  Inputs:
%    A      - Input tensor of arbitrary dimensions
%    factor - Factor matrix to multiply with (size: [new_dim x old_dim])
%    mode   - Mode/dimension along which to perform the contraction (1-indexed)
%
%  Output:
%    A      - Contracted tensor with updated size along the specified mode
%
%  The function reshapes the tensor to facilitate matrix multiplication using
%  pagemtimes for efficient computation, then reshapes back to the original
%  number of dimensions with the updated size along the contracted mode.

% Copyright 2026 The MathWorks, Inc.

    arguments
        A
        factor (:, :)
        mode (1, 1) double
    end

    sz = size(A);
    % Reshape to [prod(before) x sz(mode) x prod(after)]
    A = reshape(A, prod(sz(1:mode-1)), sz(mode), []);
    
    % Multiply along the second dimension with pagemtimes
    % new size = [prod(before) x original_size(i) x prod(after)]
    A = pagemtimes(A, 'none', factor, 'transpose');
    
    % Update size for this mode and reshape back to N-D
    sz(mode) = size(factor, 1);
    A = reshape(A, sz);
end