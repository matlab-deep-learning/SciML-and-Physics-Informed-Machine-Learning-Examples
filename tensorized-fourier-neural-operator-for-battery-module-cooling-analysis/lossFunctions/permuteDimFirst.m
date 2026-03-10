function X = permuteDimFirst(X, dim)
%PERMUTEDIMFIRST - Permute specified dimension to be the first dimension.
%   X = PERMUTEDIMFIRST(X, DIM) moves the dimension specified by DIM
%   to the first position while maintaining the relative order of other
%   dimensions.
    fmt = dims(X);
    Dim = finddim(X, dim);
    permuteOrder = [Dim setdiff(1:ndims(X), Dim, 'stable')];
    X = permute(stripdims(X), permuteOrder);
    X = dlarray(X, fmt);
end