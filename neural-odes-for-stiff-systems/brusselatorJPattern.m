function S = brusselatorJPattern(N)
%brusselatorJPattern Structural sparsity pattern of Brusselator 2D Jacobian.
%   S = brusselatorJPattern(N) returns the n-by-n structural sparsity
%   pattern (logical sparse) for the 2D Brusselator system on an N-by-N
%   grid, where n = 2*N*N.
%
%   The pattern is state-independent: it marks every entry that CAN be
%   nonzero for any state values. This is the correct input for JPattern.

nn = N*N;
n  = 2*nn;

idx = reshape(1:nn, N, N);
center = idx(:);
idxm1 = [N 1:(N-1)];
idxp1 = [2:N 1];
left   = idx(idxm1,:); left  = left(:);
right  = idx(idxp1,:); right = right(:);
down   = idx(:,idxm1); down  = down(:);
up     = idx(:,idxp1); up    = up(:);

o = ones(nn, 1);

% uu block: 5-point Laplacian stencil + diagonal reaction
uu_r = [center; center; center; center; center];
uu_c = [center; left; right; down; up];
uu_v = [o; o; o; o; o];

% uv block: diagonal coupling (u^2 term — structurally nonzero everywhere)
uv_r = center;
uv_c = nn + center;
uv_v = o;

% vu block: diagonal coupling (A - 2*u*v — structurally nonzero everywhere)
vu_r = nn + center;
vu_c = center;
vu_v = o;

% vv block: 5-point Laplacian stencil + diagonal reaction
vv_r = nn + [center; center; center; center; center];
vv_c = nn + [center; left; right; down; up];
vv_v = [o; o; o; o; o];

S = sparse([uu_r; uv_r; vu_r; vv_r], ...
           [uu_c; uv_c; vu_c; vv_c], ...
           [uu_v; uv_v; vu_v; vv_v], n, n) ~= 0;

end
