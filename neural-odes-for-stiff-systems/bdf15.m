function [tout, yout] = bdf15(odefun, tspan, y0, options)
%BDF15 Simplified stiff ODE solver with autodiff Jacobians.
%
%   [T, Y] = BDF15(ODEFUN, TSPAN, Y0) solves the ODE y' = f(t,y)
%   using variable-order (1-5) BDF/NDF methods with adaptive step size.
%   Jacobians are computed via finite differences.
%
%   [T, Y] = BDF15(ODEFUN, TSPAN, Y0, OPTIONS) uses OPTIONS from
%   odeset. Supported: RelTol, AbsTol, MaxStep, InitialStep, Jacobian,
%   MaxOrder, BDF. Other fields are ignored. Jacobian is a matrix or a
%   function handle dfdy = jacfcn(t, y), with the same arguments as ODEFUN.
%
%   Compatible with dlarray for gradient computation via dlfeval/dlgradient.
%   The full Newton iteration is traced for correct backpropagation.

if nargin < 4
    options = [];
end

rtol   = odeget(options, 'RelTol', 1e-3);
atol   = odeget(options, 'AbsTol', 1e-6);
hmax   = odeget(options, 'MaxStep', []);
htry   = odeget(options, 'InitialStep', []);
maxk   = odeget(options, 'MaxOrder', 5);
useBDF = isequal(odeget(options, 'BDF', 'off'), 'on');
Jac    = odeget(options, 'Jacobian', []);

t0     = tspan(1);
tfinal = tspan(end);
tdir   = sign(tfinal - t0);
neq    = numel(y0);
y      = y0(:);
t      = t0;
useDl  = isa(y, 'dlarray');

if isempty(hmax)
    hmax = 0.1 * abs(tfinal - t0);
end

% NDF coefficients (Klopfenstein-Shampine)
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
f0 = odefun(t, y);
dfdyNum = iComputeJacobian(odefun, Jac, t, extractdata(y), extractdata(f0));

% Initial step size (all numeric)
ypNum = extractdata(f0);
threshold = atol / rtol;
wt = max(abs(extractdata(y)), threshold);
rh = 1.25 * norm(ypNum ./ wt, inf) / sqrt(rtol);
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
f1Num = extractdata(odefun(t + tdel, y));
DfDt = (f1Num - ypNum) ./ tdel + dfdyNum * ypNum;
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

% Backward differences table: dlarray to carry autodiff trace
if useDl
    dif = dlarray(zeros(neq, maxk+2));
else
    dif = zeros(neq, maxk+2);
end
dif(:,1) = h * f0;

hinvGak = h * invGa(k);
nconhk = 0;

% Numeric iteration matrix and LU factorization
In = eye(neq);
miterNum = In - hinvGak * dfdyNum;
[Llu, Ulu, Plu] = lu(miterNum);

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
            % Traced predictor
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

            predNum = extractdata(pred);
            wt = max(max(abs(extractdata(y)), abs(predNum)), threshold);
            invwt = 1 ./ wt;
            minnrm = 100 * eps * norm(predNum .* invwt, inf);

            % Newton iteration -- fully traced for autodiff
            if useDl
                difkp1 = dlarray(zeros(neq, 1));
            else
                difkp1 = zeros(neq, 1);
            end
            ynew = pred;
            tooslow = false;
            rate = 0;

            for iter = 1:maxit
                fnew = odefun(tnew, ynew);
                rhs = hinvGak * fnew - (psi + difkp1);
                % Solve with precomputed numeric LU applied to traced rhs
                del = iLuSolve(Llu, Ulu, Plu, rhs);

                newnrm = norm(extractdata(del) .* invwt, inf);
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
                    yNum = extractdata(y);
                    fCurNum = extractdata(odefun(t, y));
                    dfdyNum = iComputeJacobian(odefun, Jac, t, yNum, fCurNum);
                    Jcurrent = true;
                    miterNum = In - hinvGak * dfdyNum;
                    [Llu, Ulu, Plu] = lu(miterNum);
                elseif absh <= hmin
                    warning('bdf15:StepSizeTooSmall', ...
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
                    hinvGak = h * invGa(k);
                    nconhk = 0;
                    miterNum = In - hinvGak * dfdyNum;
                    [Llu, Ulu, Plu] = lu(miterNum);
                end
                havrate = false;
            end
        end

        % Error estimation (numeric)
        difkp1Num = extractdata(difkp1);
        err = norm(difkp1Num .* invwt, inf) * erconst(k);

        if err > rtol
            if absh <= hmin
                warning('bdf15:StepSizeTooSmall', ...
                    'Step size fell below minimum at t = %e.', t);
                done = true;
                break;
            end

            abshlast = absh;
            if nofailed
                nofailed = false;
                hopt = absh * max(0.1, 0.833*(rtol/err)^(1/(k+1)));
                if k > 1
                    errkm1 = norm((extractdata(dif(:,k)) + difkp1Num) .* invwt, inf) * erconst(k-1);
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
            hinvGak = h * invGa(k);
            nconhk = 0;
            miterNum = In - hinvGak * dfdyNum;
            [Llu, Ulu, Plu] = lu(miterNum);
            havrate = false;
        else
            break;
        end
    end

    % Update backward differences (all traced)
    dif(:,k+2) = difkp1 - dif(:,k+1);
    dif(:,k+1) = difkp1;
    for j = k:-1:1
        dif(:,j) = dif(:,j) + dif(:,j+1);
    end

    toutList(end+1,1) = tnew; %#ok<AGROW>
    youtList(:,end+1) = ynew; %#ok<AGROW>

    % Advance
    klast = k;
    abshlast = absh;
    t = tnew;
    y = ynew;
    Jcurrent = false;

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
            errkm1 = norm(extractdata(dif(:,k)) .* invwt, inf) * erconst(k-1);
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
            errkp1 = norm(extractdata(dif(:,k+2)) .* invwt, inf) * erconst(k+1);
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

end

% Solve L*U*x = P*b with precomputed numeric LU, traced rhs b
function x = iLuSolve(L, U, P, b)
n = size(L, 1);
% Forward substitution: L*z = P*b
Pb = P * b;
z = Pb;
for i = 2:n
    z(i) = Pb(i) - L(i,1:i-1) * z(1:i-1);
end
% Back substitution: U*x = z
x = z;
for i = n:-1:1
    x(i) = (z(i) - U(i,i+1:n) * x(i+1:n)) / U(i,i);
end
end

% Jacobian via finite differences (numeric only)
function J = iComputeJacobian(odefun, Jac, t, yNum, f0Num)
if ~isempty(Jac)
    if isa(Jac, 'function_handle')
        J = Jac(t, yNum);
    else
        J = Jac;
    end
    return
end
n = numel(yNum);
sqrtEps = sqrt(eps);
J = zeros(n, n);
for j = 1:n
    ypert = yNum;
    delta = sqrtEps * max(abs(yNum(j)), 1);
    ypert(j) = ypert(j) + delta;
    J(:,j) = (odefun(t, ypert) - f0Num) / delta;
end
end

% Rescale backward differences for step/order change
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
