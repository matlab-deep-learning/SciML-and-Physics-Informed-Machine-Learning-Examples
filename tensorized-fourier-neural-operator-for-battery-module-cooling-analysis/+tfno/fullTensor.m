function Afull = fullTensor(core, factors)
% FULLTENSOR Reconstruct a full tensor from its Tucker decomposition.
%   Afull = FULLTENSOR(core, factors) reconstructs a full tensor from
%   its Tucker decomposition components. The Tucker decomposition represents
%   a tensor as a core tensor multiplied by factor matrices along each mode.
%
%   Inputs:
%     core    - N-dimensional core tensor
%     factors - Cell array of length N containing factor matrices
%               factors{i} is a matrix of size original_size(i) x size(core, i)
%
%   Output:
%     Afull   - Reconstructed full tensor
%
%   The function iteratively contracts the core tensor with each factor
%   matrix along the corresponding mode to reconstruct the original tensor.

% Copyright 2026 The MathWorks, Inc.
    
    coder.varsize('Afull');

    Afull = core;
    for mode = 1:ndims(Afull)
        Afull = tfno.contractFactor(Afull, factors{mode}, mode);
    end
end