function [u, v] = brusselatorInitialCondition(N)
% This function initializes the state variables. It sets up initial
% concentrations for both species based on parabolic-like profiles,
% ensuring non-uniform starting conditions that can lead to interesting
% dynamics.

%   Copyright 2026 The MathWorks, Inc.

xyd = linspace(0,1,N);
u = zeros(N, N);
v = zeros(N, N);
for i = 1:N
    x = xyd(i);
    for j = 1:N
        y = xyd(j);
        u(i, j) = 22*(y*(1-y))^(3/2);
        v(i, j) = 27*(x*(1-x))^(3/2);
    end
end
end