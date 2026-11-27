function [tout, yout, Sx0T, SthetaT] = bdf15sForwardSensitivity(odefun, tspan, y0, theta, options)
%bdf15sForwardSensitivity BDF solver with forward sensitivity and JPattern support.
%   [T, Y, SX0T, STHT] = bdf15sForwardSensitivity(ODEFUN, TSPAN, Y0, THETA, OPTIONS)
%
%   Solves y' = odefun(t, y, theta) and propagates forward sensitivities
%   dy/dy0 and dy/dtheta. Returns the full trajectory plus final-time
%   sensitivity matrices.
%
%   ODEFUN is a function handle with signature f = odefun(t, y, theta).
%   THETA is the parameter vector (m-by-1), or [] if there are no parameters.
%
%   OPTIONS is an odeset structure. Supported fields:
%     RelTol, AbsTol  - Error tolerances (defaults 1e-3, 1e-6)
%     MaxStep         - Maximum step size
%     InitialStep     - Suggested initial step size
%     MaxOrder        - Maximum BDF order 1-5 (default 5)
%     BDF             - 'on' for pure BDF, 'off' (default) for NDF
%     JPattern        - Sparsity pattern of df/dy (n-by-n sparse or full).
%                       Enables sparse finite-difference Jacobian via
%                       column grouping. Reduces Jacobian cost from O(n)
%                       to O(nColors) function evaluations. Also enables
%                       sparse LU for Newton solves. Ignored when an
%                       analytic Jacobian is provided.
%     Jacobian        - function handle: [dfdy, dfdtheta] = jacfcn(t, y, theta)
%                       Takes the same arguments as ODEFUN.
%
%   Other odeset fields are ignored. OPTIONS defaults to [].
%
%   Outputs:
%     T     - time points (nSteps-by-1)
%     Y     - solution (nSteps-by-n)
%     SX0T  - dy(T)/dy0 (n-by-n)
%     STHT  - dy(T)/dtheta (n-by-m)

if nargin < 5
    options = [];
end

rtol   = odeget(options, 'RelTol', 1e-3);
atol   = odeget(options, 'AbsTol', 1e-6);
hmax   = odeget(options, 'MaxStep', []);
htry   = odeget(options, 'InitialStep', []);
maxk   = odeget(options, 'MaxOrder', 5);
useBDF = isequal(odeget(options, 'BDF', 'off'), 'on');

jacUser  = iGetField(options, 'Jacobian', []);
jpattern = odeget(options, 'JPattern', []);

t0     = tspan(1);
tfinal = tspan(end);
tdir   = sign(tfinal - t0);
neq    = numel(y0);
if isempty(theta)
    thetaVal = [];
    m = 0;
else
    thetaVal = theta(:);
    m = numel(thetaVal);
end
y      = y0(:);
t      = t0;

% Closure for BDF state solve (2-arg form)
odefunClosed = @(t, y) odefun(t, y, thetaVal);

% Build Jacobian function
useSparse = false;
if ~isempty(jacUser)
    if ~isa(jacUser, 'function_handle')
        error('bdf15sForwardSensitivity:InvalidJacobian', ...
            'The Jacobian option must be a function handle [dfdy, dfdtheta] = jacfcn(t, y, theta).');
    end
    jacFcn = @(t, y) jacUser(t, y, thetaVal);
else
    if ~isempty(jpattern)
        S = sparse(jpattern) ~= 0;
        groups = iColumnGrouping(S);
        jacFcn = @(t, y) iSparseJacobianWithTheta(odefun, t, y, thetaVal, S, groups, neq, m);
        useSparse = true;
    else
        jacFcn = @(t, y) iNumericJacobians(odefun, t, y, thetaVal, neq, m);
    end
end

if isempty(hmax)
    hmax = 0.1 * abs(tfinal - t0);
end

% NDF coefficients
G = [1; 3/2; 11/6; 25/12; 137/60];
if useBDF
    alpha = [0; 0; 0; 0; 0];
else
    alpha = [-37/200; -1/9; -0.0823; -0.0415; 0];
end
invGa   = 1 ./ (G .* (1 - alpha));
erconst = alpha .* G + (1 ./ (2:6)');

difU = [-1, -2, -3, -4,  -5;
         0,  1,  3,  6,  10;
         0,  0, -1, -4, -10;
         0,  0,  0,  1,   5;
         0,  0,  0,  0,  -1];
difU = difU(1:maxk, 1:maxk);

maxit = 4;

% Initial evaluation
f0 = odefunClosed(t, y);
[dfdyNum, dfdth0] = jacFcn(t, y);

% Auto-detect sparsity from Jacobian output
if issparse(dfdyNum)
    useSparse = true;
end

if useSparse
    In = speye(neq);
else
    In = eye(neq);
end

% Initial step size
threshold = atol / rtol;
wt = max(abs(y), threshold);
rh = 1.25 * norm(f0 ./ wt, inf) / sqrt(rtol);
absh = min(hmax, abs(tfinal - t0));
if absh * rh > 1
    absh = 1 / rh;
end
hmin = 16 * eps(abs(t));
absh = max(absh, hmin);
if ~isempty(htry)
    absh = min(absh, htry);
end

% Refine initial step
h = tdir * absh;
tdel = (t + tdir * min(sqrt(eps)*max(abs(t), abs(t+h)), absh)) - t;
f1 = odefunClosed(t + tdel, y);
DfDt = (f1 - f0) ./ tdel + dfdyNum * f0;
rh = 1.25 * sqrt(0.5 * norm(DfDt ./ wt, inf) / rtol);
abshNew = min(hmax, abs(tfinal - t0));
if abshNew * rh > 1
    abshNew = 1 / rh;
end
absh = max(abshNew, hmin);
h = tdir * absh;

% Initialize BDF state
k = 1;
K = 1:k;
klast = k;
abshlast = absh;

dif = zeros(neq, maxk+2);
dif(:,1) = h * f0;

hinvGak = h * invGa(k);
nconhk = 0;

miterNum = In - hinvGak * dfdyNum;
[Llu, Ulu, Plu] = lu(miterNum);

% Initialize sensitivities
nDifCols = maxk + 2;
Sdif_x0 = cell(nDifCols, 1);
Sdif_theta = cell(nDifCols, 1);
for j = 1:nDifCols
    Sdif_x0{j} = zeros(neq, neq);
    Sdif_theta{j} = zeros(neq, m);
end
Sdif_x0{1} = h * dfdyNum;
Sdif_theta{1} = h * dfdth0;

Sx0 = eye(neq);
Stheta = zeros(neq, m);

% Trajectory storage
toutList = t;
youtList = y;

Jcurrent = true;
havrate = false;

% Main loop
done = false;
while ~done

    hmin = 16 * eps(abs(t));
    absh = min(hmax, max(hmin, absh));
    h = tdir * absh;

    if 1.1 * absh >= abs(tfinal - t)
        h = tfinal - t;
        absh = abs(h);
        done = true;
    end

    % Rescale differences if h or k changed
    if (absh ~= abshlast) || (k ~= klast)
        difRU = iComputeDifRU(k, absh/abshlast, difU(K,K));
        dif(:,K) = dif(:,K) * difRU;
        Sdif_x0(1:k) = iRescaleSdif(Sdif_x0(1:k), difRU, k, neq);
        Sdif_theta(1:k) = iRescaleSdif(Sdif_theta(1:k), difRU, k, m);
        hinvGak = h * invGa(k);
        nconhk = 0;
        miterNum = In - hinvGak * dfdyNum;
        [Llu, Ulu, Plu] = lu(miterNum);
        havrate = false;
    end

    % Advance one step
    nofailed = true;
    while true
        gotynew = false;
        while ~gotynew
            psi = dif(:,K) * (G(K) * invGa(k));
            tnew = t + h;
            if done
                tnew = tfinal;
            end
            h = tnew - t;

            pred = y;
            for i = k:-1:1
                pred = pred + dif(:,i);
            end

            wt = max(max(abs(y), abs(pred)), threshold);
            invwt = 1 ./ wt;
            minnrm = 100 * eps * norm(pred .* invwt, inf);

            difkp1 = zeros(neq, 1);
            ynew = pred;
            tooslow = false;
            rate = 0;

            for iter = 1:maxit
                fnew = odefunClosed(tnew, ynew);
                rhs = hinvGak * fnew - (psi + difkp1);
                del = Ulu \ (Llu \ (Plu * rhs));

                newnrm = norm(del .* invwt, inf);
                difkp1 = difkp1 + del;
                ynew = pred + difkp1;

                if newnrm <= minnrm
                    gotynew = true;
                    break;
                elseif iter == 1
                    if havrate
                        errit = newnrm * rate / (1 - rate);
                        if errit <= 0.05 * rtol
                            gotynew = true;
                            break;
                        end
                    else
                        rate = 0;
                    end
                elseif newnrm > 0.9 * oldnrm
                    tooslow = true;
                    break;
                else
                    rate = max(0.9*rate, newnrm / oldnrm);
                    havrate = true;
                    errit = newnrm * rate / (1 - rate);
                    if errit <= 0.5 * rtol
                        gotynew = true;
                        break;
                    elseif iter == maxit
                        tooslow = true;
                        break;
                    elseif 0.5*rtol < errit * rate^(maxit - iter)
                        tooslow = true;
                        break;
                    end
                end
                oldnrm = newnrm;
            end

            if tooslow
                if ~Jcurrent
                    [dfdyNum, ~] = jacFcn(t, y);
                    Jcurrent = true;
                    miterNum = In - hinvGak * dfdyNum;
                    [Llu, Ulu, Plu] = lu(miterNum);
                elseif absh <= hmin
                    warning('bdf15sForwardSensitivity:StepSizeTooSmall', ...
                        'Step size fell below minimum at t = %e.', t);
                    done = true;
                    gotynew = true;
                else
                    abshlast = absh;
                    absh = max(0.3 * absh, hmin);
                    h = tdir * absh;
                    done = false;
                    difRU = iComputeDifRU(k, absh/abshlast, difU(K,K));
                    dif(:,K) = dif(:,K) * difRU;
                    Sdif_x0(1:k) = iRescaleSdif(Sdif_x0(1:k), difRU, k, neq);
                    Sdif_theta(1:k) = iRescaleSdif(Sdif_theta(1:k), difRU, k, m);
                    hinvGak = h * invGa(k);
                    nconhk = 0;
                    miterNum = In - hinvGak * dfdyNum;
                    [Llu, Ulu, Plu] = lu(miterNum);
                end
                havrate = false;
            end
        end

        % Error estimation
        err = norm(difkp1 .* invwt, inf) * erconst(k);

        if err > rtol
            if absh <= hmin
                warning('bdf15sForwardSensitivity:StepSizeTooSmall', ...
                    'Step size fell below minimum at t = %e.', t);
                done = true;
                break;
            end

            abshlast = absh;
            if nofailed
                nofailed = false;
                hopt = absh * max(0.1, 0.833*(rtol/err)^(1/(k+1)));
                if k > 1
                    errkm1 = norm((dif(:,k) + difkp1) .* invwt, inf) * erconst(k-1);
                    hkm1 = absh * max(0.1, 0.769*(rtol/errkm1)^(1/k));
                    if hkm1 > hopt
                        hopt = min(absh, hkm1);
                        k = k - 1;
                        K = 1:k;
                    end
                end
                absh = max(hmin, hopt);
            else
                absh = max(hmin, 0.5 * absh);
            end
            h = tdir * absh;
            if absh < abshlast
                done = false;
            end

            difRU = iComputeDifRU(k, absh/abshlast, difU(K,K));
            dif(:,K) = dif(:,K) * difRU;
            Sdif_x0(1:k) = iRescaleSdif(Sdif_x0(1:k), difRU, k, neq);
            Sdif_theta(1:k) = iRescaleSdif(Sdif_theta(1:k), difRU, k, m);
            hinvGak = h * invGa(k);
            nconhk = 0;
            miterNum = In - hinvGak * dfdyNum;
            [Llu, Ulu, Plu] = lu(miterNum);
            havrate = false;
        else
            break;
        end
    end

    % === Sensitivity propagation at converged step ===
    [dfdyNew, dfdthNew] = jacFcn(tnew, ynew);

    hinvGakNew = h * invGa(k);
    MiterNew = In - hinvGakNew * dfdyNew;

    GKcoeffs = G(K) * invGa(k);

    Spred_x0 = Sx0;
    Spsi_x0 = zeros(neq, neq);
    for j = 1:k
        Spred_x0 = Spred_x0 + Sdif_x0{j};
        Spsi_x0 = Spsi_x0 + GKcoeffs(j) * Sdif_x0{j};
    end
    Sx0_new = MiterNew \ (Spred_x0 - Spsi_x0);

    Spred_th = Stheta;
    Spsi_th = zeros(neq, m);
    for j = 1:k
        Spred_th = Spred_th + Sdif_theta{j};
        Spsi_th = Spsi_th + GKcoeffs(j) * Sdif_theta{j};
    end
    if m > 0
        Stheta_new = MiterNew \ (Spred_th - Spsi_th + hinvGakNew * dfdthNew);
    else
        Stheta_new = zeros(neq, 0);
    end

    % Update sensitivity difference tables
    Sdifkp1_x0 = Sx0_new - Spred_x0;
    Sdif_x0{k+2} = Sdifkp1_x0 - Sdif_x0{k+1};
    Sdif_x0{k+1} = Sdifkp1_x0;
    for j = k:-1:1
        Sdif_x0{j} = Sdif_x0{j} + Sdif_x0{j+1};
    end

    if m > 0
        Sdifkp1_th = Stheta_new - Spred_th;
        Sdif_theta{k+2} = Sdifkp1_th - Sdif_theta{k+1};
        Sdif_theta{k+1} = Sdifkp1_th;
        for j = k:-1:1
            Sdif_theta{j} = Sdif_theta{j} + Sdif_theta{j+1};
        end
    end

    Sx0 = Sx0_new;
    Stheta = Stheta_new;

    % Update backward differences (state)
    dif(:,k+2) = difkp1 - dif(:,k+1);
    dif(:,k+1) = difkp1;
    for j = k:-1:1
        dif(:,j) = dif(:,j) + dif(:,j+1);
    end

    % Store trajectory
    toutList(end+1,1) = tnew; %#ok<AGROW>
    youtList(:,end+1) = ynew; %#ok<AGROW>

    % Advance
    klast = k;
    abshlast = absh;
    t = tnew;
    y = ynew;
    dfdyNum = dfdyNew;
    Jcurrent = true;

    % Order and step size selection
    nconhk = min(nconhk + 1, maxk + 2);
    if nconhk >= k + 2
        temp = 1.2 * (err/rtol)^(1/(k+1));
        if temp > 0.1
            hopt = absh / temp;
        else
            hopt = 10 * absh;
        end
        kopt = k;

        if k > 1
            errkm1 = norm(dif(:,k) .* invwt, inf) * erconst(k-1);
            temp = 1.3 * (errkm1/rtol)^(1/k);
            if temp > 0.1
                hkm1 = absh / temp;
            else
                hkm1 = 10 * absh;
            end
            if hkm1 > hopt
                hopt = hkm1;
                kopt = k - 1;
            end
        end

        if k < maxk
            errkp1 = norm(dif(:,k+2) .* invwt, inf) * erconst(k+1);
            temp = 1.4 * (errkp1/rtol)^(1/(k+2));
            if temp > 0.1
                hkp1 = absh / temp;
            else
                hkp1 = 10 * absh;
            end
            if hkp1 > hopt
                hopt = hkp1;
                kopt = k + 1;
            end
        end

        if hopt > absh
            absh = hopt;
            if k ~= kopt
                k = kopt;
                K = 1:k;
            end
        end
    end
end

tout = toutList;
yout = youtList';
Sx0T = Sx0;
SthetaT = Stheta;

end

%% ---- JPattern helpers ----

function groups = iColumnGrouping(S)
%iColumnGrouping Greedy graph coloring for sparse Jacobian column grouping.
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

function [dfdx, dfdtheta] = iSparseJacobianWithTheta(odefun, t, y, theta, S, groups, n, m)
%iSparseJacobianWithTheta Sparse FD for df/dy via column grouping, dense FD for df/dtheta.
sqrtEps = sqrt(eps);
f0 = odefun(t, y, theta);
nGroups = numel(groups);
totalNnz = nnz(S);

% --- df/dy via column grouping ---
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

% --- df/dtheta via dense column-by-column FD ---
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

%% ---- Standard helpers ----

function Sdif = iRescaleSdif(Sdif, difRU, k, ~)
newSdif = cell(size(Sdif));
for j = 1:k
    newSdif{j} = zeros(size(Sdif{1}));
    for i = 1:k
        newSdif{j} = newSdif{j} + Sdif{i} * difRU(i,j);
    end
end
Sdif = newSdif;
end

function [dfdx, dfdtheta] = iNumericJacobians(odefun, t, y, theta, n, m)
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

function difRU = iComputeDifRU(k, hratio, difUK)
difRU = zeros(k, k);
for j = 1:k
    difRU(1,j) = -j * hratio;
    for i = 2:k
        difRU(i,j) = ((i - 1 - j*hratio) / i) * difRU(i-1,j);
    end
end
difRU = difRU * difUK;
end

function v = iGetField(s, name, default)
if isstruct(s) && isfield(s, name)
    v = s.(name);
else
    v = default;
end
end
