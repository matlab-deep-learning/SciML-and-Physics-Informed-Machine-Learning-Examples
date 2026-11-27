%% Performance Comparison: Brusselator 2D Gradient Methods
% Compares wall-clock time for gradient computation at increasing grid sizes:
%   1. ode15sForwardSensitivity — dense FD Jacobian
%   2. ode15sForwardSensitivity — JPattern sparse FD Jacobian
%   3. ode15sForwardSensitivity — analytic Jacobian
%   4. ode15sAdjoint            — dense FD Jacobian (adjoint ODE via ode15s)
%   5. ode15sAdjoint            — JPattern sparse FD Jacobian
%   6. ode15sAdjoint            — analytic Jacobian
%   7. bdf15                    — dlarray reverse-mode AD

fprintf('==========================================================\n');
fprintf(' Brusselator 2D Performance Comparison\n');
fprintf('==========================================================\n\n');

gridSizes = [4, 8, 12];
tspan = [0 1];
options = odeset(RelTol=1e-4, AbsTol=1e-6);

A = 3.4;
B = 1;
alpha = 10;

nWarmup = 1;
nRuns = 3;

nMethods = 7;
methodNames = { ...
    'FwdSens (dense FD)', ...
    'FwdSens (JPattern)', ...
    'FwdSens (analytic)', ...
    'Adjoint (dense FD)', ...
    'Adjoint (JPattern)', ...
    'Adjoint (analytic)', ...
    'bdf15'};

allTimes = zeros(numel(gridSizes), nMethods);
stateDims = zeros(numel(gridSizes), 1);

for gi = 1:numel(gridSizes)
    N = gridSizes(gi);
    n = 2*N*N;
    stateDims(gi) = n;
    fprintf('--- N=%d (state dim = %d) ---\n', N, n);

    [u0, v0] = brusselatorInitialCondition(N);
    state0 = [u0(:); v0(:)];
    xd = linspace(0, 1, N);
    param = [A; B; alpha; xd(:)];

    x0dl = dlarray(state0);
    thetadl = dlarray(param);

    Jpat = brusselatorJPattern(N);
    nColors = iCountColors(Jpat);
    fprintf('  JPattern: nnz=%d, nColors=%d (vs n=%d dense FD columns)\n', ...
        nnz(Jpat), nColors, n);

    optionsJP = options;
    optionsJP.JPattern = Jpat;
    optionsAna = odeset(options, Jacobian=@brusselatorJacobian);

    %% 1. Forward Sensitivity — Dense FD
    s = ode15sForwardSensitivity(@brusselatorSystem, tspan, options);
    for w = 1:nWarmup; dlfeval(@iLossGrad, s, x0dl, thetadl); end
    tG = zeros(nRuns, 1);
    for r = 1:nRuns; tic; [~, gx_fs_d, gth_fs_d] = dlfeval(@iLossGrad, s, x0dl, thetadl); tG(r) = toc; end
    allTimes(gi, 1) = median(tG);

    %% 2. Forward Sensitivity — JPattern
    s = ode15sForwardSensitivity(@brusselatorSystem, tspan, optionsJP);
    for w = 1:nWarmup; dlfeval(@iLossGrad, s, x0dl, thetadl); end
    tG = zeros(nRuns, 1);
    for r = 1:nRuns; tic; dlfeval(@iLossGrad, s, x0dl, thetadl); tG(r) = toc; end
    allTimes(gi, 2) = median(tG);

    %% 3. Forward Sensitivity — Analytic
    s = ode15sForwardSensitivity(@brusselatorSystem, tspan, optionsAna);
    for w = 1:nWarmup; dlfeval(@iLossGrad, s, x0dl, thetadl); end
    tG = zeros(nRuns, 1);
    for r = 1:nRuns; tic; dlfeval(@iLossGrad, s, x0dl, thetadl); tG(r) = toc; end
    allTimes(gi, 3) = median(tG);

    %% 4. Adjoint ODE — Dense FD
    s = ode15sAdjoint(@brusselatorSystem, tspan, options);
    for w = 1:nWarmup; dlfeval(@iLossGrad, s, x0dl, thetadl); end
    tG = zeros(nRuns, 1);
    for r = 1:nRuns; tic; [~, gx_adj_d, gth_adj_d] = dlfeval(@iLossGrad, s, x0dl, thetadl); tG(r) = toc; end
    allTimes(gi, 4) = median(tG);

    %% 5. Adjoint ODE — JPattern
    s = ode15sAdjoint(@brusselatorSystem, tspan, optionsJP);
    for w = 1:nWarmup; dlfeval(@iLossGrad, s, x0dl, thetadl); end
    tG = zeros(nRuns, 1);
    for r = 1:nRuns; tic; dlfeval(@iLossGrad, s, x0dl, thetadl); tG(r) = toc; end
    allTimes(gi, 5) = median(tG);

    %% 6. Adjoint ODE — Analytic
    s = ode15sAdjoint(@brusselatorSystem, tspan, optionsAna);
    for w = 1:nWarmup; dlfeval(@iLossGrad, s, x0dl, thetadl); end
    tG = zeros(nRuns, 1);
    for r = 1:nRuns; tic; dlfeval(@iLossGrad, s, x0dl, thetadl); tG(r) = toc; end
    allTimes(gi, 6) = median(tG);

    %% 7. bdf15 (dlarray tracing)
    paramDl = dlarray(param);
    x0dl2 = dlarray(state0);
    for w = 1:nWarmup; dlfeval(@iLossGradDlarray, x0dl2, paramDl, tspan, options); end
    tG = zeros(nRuns, 1);
    for r = 1:nRuns; tic; [~, gx_dl, gth_dl] = dlfeval(@iLossGradDlarray, x0dl2, paramDl, tspan, options); tG(r) = toc; end
    allTimes(gi, 7) = median(tG);

    %% Report
    fprintf('  %-36s %10s\n', 'Method', 'Gradient');
    fprintf('  %-36s %9.4fs\n', 'ode15sForwardSensitivity (dense FD)',  allTimes(gi, 1));
    fprintf('  %-36s %9.4fs\n', 'ode15sForwardSensitivity (JPattern)',  allTimes(gi, 2));
    fprintf('  %-36s %9.4fs\n', 'ode15sForwardSensitivity (analytic)',  allTimes(gi, 3));
    fprintf('  %-36s %9.4fs\n', 'ode15sAdjoint (dense FD)',      allTimes(gi, 4));
    fprintf('  %-36s %9.4fs\n', 'ode15sAdjoint (JPattern)',      allTimes(gi, 5));
    fprintf('  %-36s %9.4fs\n', 'ode15sAdjoint (analytic)',      allTimes(gi, 6));
    fprintf('  %-36s %9.4fs\n', 'bdf15',                         allTimes(gi, 7));

    fprintf('  Speedup vs bdf15:\n');
    fprintf('    FwdSens dense:    %5.1fx\n', allTimes(gi, 7) / allTimes(gi, 1));
    fprintf('    FwdSens JPattern: %5.1fx\n', allTimes(gi, 7) / allTimes(gi, 2));
    fprintf('    FwdSens analytic: %5.1fx\n', allTimes(gi, 7) / allTimes(gi, 3));
    fprintf('    Adjoint dense:    %5.1fx\n', allTimes(gi, 7) / allTimes(gi, 4));
    fprintf('    Adjoint JPattern: %5.1fx\n', allTimes(gi, 7) / allTimes(gi, 5));
    fprintf('    Adjoint analytic: %5.1fx\n', allTimes(gi, 7) / allTimes(gi, 6));

    err = @(gx, gth) [max(abs(extractdata(gx_fs_d) - extractdata(gx))), ...
                       max(abs(extractdata(gth_fs_d) - extractdata(gth)))];
    fprintf('  Gradient agreement vs FwdSens (dense FD):\n');
    e = err(gx_adj_d, gth_adj_d);  fprintf('    Adjoint dense:    max|dx0|=%.2e  max|dth|=%.2e\n', e(1), e(2));
    e = err(gx_dl, gth_dl);        fprintf('    bdf15:            max|dx0|=%.2e  max|dth|=%.2e\n', e(1), e(2));
    fprintf('\n');
end

fprintf('==========================================================\n');
fprintf(' Benchmark complete.\n');
fprintf('==========================================================\n');

%% Bar chart — Gradient computation time by method and grid size

gridLabels = "N=" + string(gridSizes(:)) + " (n=" + string(stateDims(:)) + ")";

figure(Name='Gradient Computation Time', Position=[100 100 900 500]);
bar(1:numel(gridSizes), allTimes);
set(gca, XTick=1:numel(gridSizes), XTickLabel=gridLabels);
ylabel('Median wall-clock time (s)');
xlabel('Grid size');
title('Gradient Computation Time — Brusselator 2D');
legend(methodNames, Location='eastoutside');
grid on

%% Bar chart — Speedup relative to bdf15

speedup = allTimes(:, 7) ./ allTimes(:, 1:6);
speedupNames = methodNames(1:6);

figure(Name='Speedup vs bdf15', Position=[100 650 900 500]);
bar(1:numel(gridSizes), speedup);
set(gca, XTick=1:numel(gridSizes), XTickLabel=gridLabels);
ylabel('Speedup (x)');
xlabel('Grid size');
title('Speedup Relative to bdf15 — Brusselator 2D');
legend(speedupNames, Location='eastoutside');
grid on

%% Helper functions

function [loss, gx0, gtheta] = iLossGrad(solver, x0, theta)
    [~, yout] = solver(x0, theta);
    loss = sum(yout(end,:));
    [gx0, gtheta] = dlgradient(loss, x0, theta);
end

function [loss, gx0, gtheta] = iLossGradDlarray(x0, param, tspan, options)
    odefun = @(t, y) brusselatorSystemDlarray(t, y, param);
    [~, yAll] = bdf15(odefun, tspan, x0, options);
    xt = yAll(end, :)';
    loss = sum(xt);
    [gx0, gtheta] = dlgradient(loss, x0, param);
end

function dstate = brusselatorSystemDlarray(t, state, param)
    N22 = numel(state);
    N = sqrt(N22/2);
    A = param(1);
    B = param(2);
    alpha_ = param(3);
    xyd = param(4:end);
    state = reshape(state, N, N, 2);
    dx = 1/(N-1);
    idxm1 = [N 1:(N-1)];
    idxp1 = [2:N 1];
    u2v = state(:,:,1).^2 .* state(:,:,2);
    du = alpha_ .* (state(idxm1,:,1) + state(idxp1,:,1) + state(:,idxm1,1) + state(:,idxp1,1) - 4.*state(:,:,1)) ./ (dx^2) + ...
        B + u2v - (A + 1).*state(:,:,1) + brusselator_f_dl(extractdata(xyd)', extractdata(xyd), t);
    dv = alpha_ .* (state(idxm1,:,2) + state(idxp1,:,2) + state(:,idxm1,2) + state(:,idxp1,2) - 4.*state(:,:,2)) ./ (dx^2) + ...
        A.*state(:,:,1) - u2v;
    dstate = [du(:); dv(:)];
end

function out = brusselator_f_dl(x, y, t)
    out = (((x - 0.3).^2 + (y - 0.6).^2) <= 0.1.^2) .* (t >= 1.1) .* 5;
end

function nColors = iCountColors(S)
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
end
