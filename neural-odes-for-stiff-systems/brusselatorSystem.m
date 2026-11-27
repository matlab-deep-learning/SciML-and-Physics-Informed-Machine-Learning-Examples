function dstate = brusselatorSystem(t, state, param)
% Core function that updates the Brusselator model over time using a 2D
% loop.
% State variables: The system has two chemical species, typically denoted
% as u and v.
% The function implements the discretized reaction-diffusion equations for
% both species.

%   Copyright 2026 The MathWorks, Inc.

N22 = numel(state);
N = sqrt(N22/2);
A = param(1);
B = param(2);
alpha = param(3);
xyd = param(4:end);
state = reshape(state, N, N, 2);
dx = 1/(N-1);
idxm1 = [N 1:(N-1)];
idxp1 = [2:N 1];
u2v = state(:,:,1).^2.*state(:,:,2);
du = alpha.*(state(idxm1,:,1) + state(idxp1,:,1) + state(:,idxm1,1) + state(:,idxp1,1) - 4.*state(:,:,1))./(dx^2) + ...
    B + u2v - (A + 1).*state(:,:,1) + brusselator_f(xyd', xyd, t);
dv = alpha.*(state(idxm1,:,2) + state(idxp1,:,2) + state(:,idxm1,2) + state(:,idxp1,2) - 4.*state(:,:,2))./(dx^2) + ...
    A.*state(:,:,1) - u2v;
dstate = [du(:);dv(:)];
end

function out = brusselator_f(x, y, t)
% This function defines an external forcing term. It introduces a localized
% perturbation (a "hotspot") at a specific spatial location (0.3, 0.6) and
% time t >= 1.1, where the value becomes 5.0; otherwise, it's 0. This term
% drives the system away from equilibrium.
out = (((x - 0.3).^2 + (y - 0.6).^2) <= 0.1.^2) .* (t >= 1.1) .* 5;
end