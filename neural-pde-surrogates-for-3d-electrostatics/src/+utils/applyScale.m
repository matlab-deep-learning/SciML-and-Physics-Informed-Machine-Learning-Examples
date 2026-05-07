function out = applyScale(data,offset,scale)
    % applyScale Normalize data using precomputed offset and scale.
    %
    %   out = utils.applyScale(data, offset, scale) computes
    %   (data - offset) ./ scale for each element. data can be a matrix or a
    %   cell array of matrices.
    %
    %   Inputs:
    %       data   - Matrix (D-by-N) or cell array of matrices to normalize.
    %       offset - D-by-1 offset vector.
    %       scale  - D-by-1 scale vector.
    %
    %   Output:
    %       out - Normalized data, same type as input.

    %   Copyright 2026 The MathWorks, Inc.
    
    rowMap = @(X) (X-offset)./scale;
    
    if iscell(data)
        out = cellfun(@(X) rowMap(X), data, 'UniformOutput', false);
    else
        out = rowMap(data);
    end
end