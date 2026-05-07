function out = applyInverseScale(data,offset,scale)
    % applyInverseScale Reverse normalization using precomputed offset and scale.
    %
    %   out = utils.applyInverseScale(data, offset, scale) computes
    %   data .* scale + offset for each element. data can be a matrix or a
    %   cell array of matrices.
    %
    %   Inputs:
    %       data   - Matrix (D-by-N) or cell array of normalized matrices.
    %       offset - D-by-1 offset vector.
    %       scale  - D-by-1 scale vector.
    %
    %   Output:
    %       out - Denormalized data, same type as input.

    %   Copyright 2026 The MathWorks, Inc.

    rowMap = @(X) X.*scale + offset;
    
    if iscell(data)
        out = cellfun(@(X) rowMap(X), data, 'UniformOutput', false);
    else
        out = rowMap(data);
    end 
end