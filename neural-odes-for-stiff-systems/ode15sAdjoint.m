classdef ode15sAdjoint < deep.DifferentiableFunction
% ode15sAdjoint Differentiable stiff ODE solver using continuous adjoint method.
%
%   solver = ode15sAdjoint(odefun, tspan)
%   solver = ode15sAdjoint(odefun, tspan, odeOpts)
%
%   Then call:
%     [tout, yout] = solver(x0, theta)
%
%   The forward pass uses MATLAB's ode15s with zero sensitivity overhead.
%   The backward pass solves the continuous adjoint ODE backward in time
%   using ode15s, following the approach in Chen et al. (2018) "Neural
%   Ordinary Differential Equations".
%
%   The adjoint state a(t) satisfies:
%     da/dt = -(df/dy)^T * a,    a(T) = dL/dy(T)
%   and the parameter gradient is accumulated as:
%     dp/dt = -(df/dtheta)^T * a,  p(T) = 0
%   integrated backward from T to t0.
%   Then dL/dy0 = a(t0), dL/dtheta = p(t0).
%
%   The augmented backward system is linear in [a; p], so Newton converges
%   in one iteration per step. The per-step cost is O(n+m) with sparse
%   Jacobians, vs O(n^2) for forward sensitivity.
%
%   odeOpts is an odeset struct. All fields are passed to ode15s for the
%   forward solve. The backward (adjoint) solve uses RelTol, AbsTol, and:
%     Jacobian  - function handle: [dfdy, dfdtheta] = jacfcn(t, y, theta)
%     JPattern  - sparsity pattern of df/dy (n-by-n sparse). Ignored when
%                 Jacobian is set.
%   Without Jacobian or JPattern, df/dy is computed by dense finite
%   differences.

    properties
        OdeFun
        Tspan
        Options
    end

    methods
        function obj = ode15sAdjoint(odefun, tspan, odeOpts)
            arguments
                odefun function_handle
                tspan (1,:)
                odeOpts = []
            end
            % 2 outputs (tout, yout), 2 memory values (trajectory t and y)
            obj@deep.DifferentiableFunction(2, ...
                NumMemoryValues=2, ...
                SaveInputsForBackward=true, ...
                SaveOutputsForBackward=false);
            obj.OdeFun = odefun;
            obj.Tspan = tspan;
            obj.Options = odeOpts;
        end

        function [tout, yout, mem_t, mem_y] = forward(obj, x0, theta)
            
            x0Val = extractdata(x0);
            if isempty(theta)
                thetaVal = [];
            else
                thetaVal = extractdata(theta);
            end

            t0 = obj.Tspan(1);
            tf = obj.Tspan(end);
            odefunClosed = @(t, y) obj.OdeFun(t, y, thetaVal);

            fwdOpts = iMakeFwdOpts(obj.Options, thetaVal);

            [tout, yout] = ode15s(odefunClosed, [t0; tf], x0Val, fwdOpts);

            mem_t = tout;
            mem_y = yout;
        end

        function [dLdx0, dLdtheta] = backward(obj, dLdtout, dLdyout, computeGradients, x0, theta, mem_t, mem_y)
            x0Val = extractdata(x0);
            if isempty(theta)
                thetaVal = [];
                m = 0;
            else
                thetaVal = extractdata(theta);
                m = numel(thetaVal);
            end
            n = numel(x0Val);

            dLdyoutVal = extractdata(dLdyout);
            aT = dLdyoutVal(end, :)';

            t0 = obj.Tspan(1);
            tf = obj.Tspan(end);

            jacFcn = iBuildJacFcn(obj.OdeFun, obj.Options, thetaVal, n, m);

            yInterp = @(t) interp1(mem_t, mem_y, t, 'pchip')';

            augRHS = @(t, z) iAdjointRHS(t, z, yInterp, jacFcn, n, m);
            augJac = @(t, z) iAugJacobian(t, z, yInterp, jacFcn, n, m);

            rtol = 1e-3; atol = 1e-6;
            if ~isempty(obj.Options)
                rtol = odeget(obj.Options, 'RelTol', rtol);
                atol = odeget(obj.Options, 'AbsTol', atol);
            end
            bwdOpts = odeset(RelTol=rtol, AbsTol=atol, ...
                Jacobian=augJac);

            z0 = [aT; zeros(m, 1)];
            [~, zout] = ode15s(augRHS, [tf, t0], z0, bwdOpts);
            zfinal = zout(end, :)';

            if computeGradients(1)
                dLdx0 = zfinal(1:n);
            else
                dLdx0 = zeros(size(x0Val));
            end

            if computeGradients(2)
                dLdtheta = zfinal(n+1:n+m);
            else
                if isempty(theta)
                    dLdtheta = [];
                else
                    dLdtheta = zeros(size(thetaVal));
                end
            end
        end
    end
end

%% Local helpers

function fwdOpts = iMakeFwdOpts(userOpts, thetaVal)
if isempty(userOpts)
    fwdOpts = odeset();
    return;
end
fwdOpts = userOpts;
if isfield(fwdOpts, 'Jacobian') && isa(fwdOpts.Jacobian, 'function_handle')
    jacRaw = fwdOpts.Jacobian;
    fwdOpts.Jacobian = @(t, y) iJacYOnly(jacRaw, t, y, thetaVal);
end
end

function Jy = iJacYOnly(jacRaw, t, y, thetaVal)
[Jy, ~] = jacRaw(t, y, thetaVal);
end

function jacFcn = iBuildJacFcn(odefun, options, thetaVal, n, m)
hasJac = ~isempty(options) && isfield(options, 'Jacobian') && ...
    isa(options.Jacobian, 'function_handle');
jpat = [];
if ~isempty(options)
    jpat = odeget(options, 'JPattern', []);
end

if hasJac
    jacRaw = options.Jacobian;
    jacFcn = @(t, y) jacRaw(t, y, thetaVal);
elseif ~isempty(jpat)
    S = sparse(jpat) ~= 0;
    groups = iColumnGrouping(S);
    jacFcn = @(t, y) iSparseJacWithTheta(odefun, t, y, thetaVal, S, groups, n, m);
else
    jacFcn = @(t, y) iDenseJac(odefun, t, y, thetaVal, n, m);
end
end

function dzdt = iAdjointRHS(t, z, yInterp, jacFcn, n, m)
a = z(1:n);
y = yInterp(t);
[Jy, Jth] = jacFcn(t, y);
dadt = -(Jy' * a);
if m > 0
    dpdt = -(Jth' * a);
    dzdt = [dadt; dpdt];
else
    dzdt = dadt;
end
end

function Jaug = iAugJacobian(t, z, yInterp, jacFcn, n, m)
y = yInterp(t);
[Jy, Jth] = jacFcn(t, y);
if m == 0
    Jaug = -Jy';
elseif issparse(Jy)
    Jaug = [-Jy', sparse(n, m); sparse(-Jth'), sparse(m, m)];
else
    Jaug = [-Jy', zeros(n, m); -Jth', zeros(m, m)];
end
end

function [dfdx, dfdtheta] = iDenseJac(odefun, t, y, theta, n, m)
sqrtEps = sqrt(eps);
f0 = odefun(t, y, theta);
dfdx = zeros(n, n);
for j = 1:n
    yp = y;
    delta = sqrtEps * max(abs(y(j)), 1);
    yp(j) = yp(j) + delta;
    dfdx(:, j) = (odefun(t, yp, theta) - f0) / delta;
end
if m == 0
    dfdtheta = zeros(n, 0);
else
    dfdtheta = zeros(n, m);
    for j = 1:m
        thp = theta;
        delta = sqrtEps * max(abs(theta(j)), 1);
        thp(j) = thp(j) + delta;
        dfdtheta(:, j) = (odefun(t, y, thp) - f0) / delta;
    end
end
end

function [dfdx, dfdtheta] = iSparseJacWithTheta(odefun, t, y, theta, S, groups, n, m)
sqrtEps = sqrt(eps);
f0 = odefun(t, y, theta);
nGroups = numel(groups);
totalNnz = nnz(S);

rows = zeros(totalNnz, 1);
cols = zeros(totalNnz, 1);
vals = zeros(totalNnz, 1);
idx = 0;
for g = 1:nGroups
    grpCols = groups{g};
    nCols = numel(grpCols);
    yp = y;
    deltas = zeros(nCols, 1);
    for c = 1:nCols
        j = grpCols(c);
        deltas(c) = sqrtEps * max(abs(y(j)), 1);
        yp(j) = yp(j) + deltas(c);
    end
    fp = odefun(t, yp, theta);
    df = fp - f0;
    for c = 1:nCols
        j = grpCols(c);
        rj = find(S(:, j));
        nR = numel(rj);
        rows(idx+1:idx+nR) = rj;
        cols(idx+1:idx+nR) = j;
        vals(idx+1:idx+nR) = df(rj) / deltas(c);
        idx = idx + nR;
    end
end
dfdx = sparse(rows, cols, vals, n, n);

if m == 0
    dfdtheta = zeros(n, 0);
else
    dfdtheta = zeros(n, m);
    for j = 1:m
        thp = theta;
        delta = sqrtEps * max(abs(theta(j)), 1);
        thp(j) = thp(j) + delta;
        dfdtheta(:, j) = (odefun(t, y, thp) - f0) / delta;
    end
end
end

function groups = iColumnGrouping(S)
n = size(S, 2);
C = (S' * S) ~= 0;
colors = zeros(n, 1);
nColors = 0;
for j = 1:n
    neighbors = find(C(:, j));
    forbidden = false(nColors + 1, 1);
    for nb = neighbors'
        if colors(nb) > 0
            forbidden(colors(nb)) = true;
        end
    end
    c = find(~forbidden, 1);
    colors(j) = c;
    if c > nColors
        nColors = c;
    end
end
groups = cell(nColors, 1);
for g = 1:nColors
    groups{g} = find(colors == g);
end
end
