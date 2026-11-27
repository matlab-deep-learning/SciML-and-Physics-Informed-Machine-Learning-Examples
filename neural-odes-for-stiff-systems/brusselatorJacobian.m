function [Jy, Jp] = brusselatorJacobian(t, state, param)
%brusselatorJacobian Analytic Jacobian for 2D Brusselator system.
%   [Jy, Jp] = brusselatorJacobian(t, state, param)
%
%   Compatible with the Jacobian option of ode15sAdjoint,
%   ode15sForwardSensitivity, and bdf15sForwardSensitivity.
%   Signature: jacfun(t, state, param) -> [dfdstate, dfdparam]
%
%   Inputs:
%     t     - current time
%     state - state vector [u(:); v(:)], length 2*N*N
%     param - parameter vector [A; B; alpha; xd(:)]
%
%   Outputs:
%     Jy - df/dstate  (2*N*N x 2*N*N)
%     Jp - df/dparam  (2*N*N x numel(param))

N22 = numel(state);
N = sqrt(N22/2);
nn = N*N;
n  = 2*nn;
A  = param(1);
alpha_ = param(3);
dx  = 1/(N-1);
dx2 = dx^2;
coeff = alpha_ / dx2;

uv = reshape(state, N, N, 2);
u  = uv(:,:,1);
v  = uv(:,:,2);

idxm1 = [N 1:(N-1)];
idxp1 = [2:N 1];

idx = reshape(1:nn, N, N);
center = idx(:);
left   = idx(idxm1,:); left  = left(:);
right  = idx(idxp1,:); right = right(:);
down   = idx(:,idxm1); down  = down(:);
up     = idx(:,idxp1); up    = up(:);

o = ones(nn, 1);

% --- df/dstate (Jy) ---
% du/du block: diffusion Laplacian + reaction terms 2*u*v - (A+1)
uu_r = [center; center; center; center; center];
uu_c = [center; left; right; down; up];
uu_v = [2*u(:).*v(:) - (A+1) - 4*coeff; coeff*o; coeff*o; coeff*o; coeff*o];

% du/dv block: reaction term u^2
uv_r = center;
uv_c = nn + center;
uv_v = u(:).^2;

% dv/du block: reaction term A - 2*u*v
vu_r = nn + center;
vu_c = center;
vu_v = A - 2*u(:).*v(:);

% dv/dv block: diffusion Laplacian + reaction term -u^2
vv_r = nn + [center; center; center; center; center];
vv_c = nn + [center; left; right; down; up];
vv_v = [-u(:).^2 - 4*coeff; coeff*o; coeff*o; coeff*o; coeff*o];

Jy = sparse([uu_r; uv_r; vu_r; vv_r], ...
             [uu_c; uv_c; vu_c; vv_c], ...
             [uu_v; uv_v; vu_v; vv_v], n, n);

% --- df/dparam (Jp) ---
Lap_u = u(idxm1,:) + u(idxp1,:) + u(:,idxm1) + u(:,idxp1) - 4*u;
Lap_v = v(idxm1,:) + v(idxp1,:) + v(:,idxm1) + v(:,idxp1) - 4*v;

% df/dA: du_ij/dA = -u_ij, dv_ij/dA = u_ij
dfdA = [-u(:); u(:)];

% df/dB: du_ij/dB = 1, dv_ij/dB = 0
dfdB = [ones(nn,1); zeros(nn,1)];

% df/dalpha: du/dalpha = Lap(u)/dx^2, dv/dalpha = Lap(v)/dx^2
dfdalpha = [Lap_u(:)/dx2; Lap_v(:)/dx2];

% df/dxd: forcing term is indicator function, derivative is zero a.e.
dfdxd = zeros(n, numel(param) - 3);

Jp = [dfdA, dfdB, dfdalpha, dfdxd];

end
