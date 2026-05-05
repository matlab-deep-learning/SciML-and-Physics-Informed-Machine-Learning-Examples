function X = permuteDimFirst(X)
%PERMUTEDIMFIRST Permute a labeled dlarray to [S1, ..., SD, C, B] physical order.
%   X = PERMUTEDIMFIRST(X) reorders the underlying data so that spatial
%   dimensions come first, followed by the channel dimension, then batch.
%   The original format labels are preserved on the output dlarray.

% Copyright 2026 The MathWorks, Inc.

    fmt = dims(X);
    sdims = finddim(X, 'S');
    cdim = finddim(X, 'C');
    bdim = finddim(X, 'B');
    permuteOrder = [sdims, cdim, bdim];
    X = permute(stripdims(X), permuteOrder);
    X = dlarray(X, fmt);
end
