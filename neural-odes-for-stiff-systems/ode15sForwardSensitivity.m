classdef ode15sForwardSensitivity < deep.DifferentiableFunction
% ode15sForwardSensitivity Differentiable stiff ODE solver using deep.DifferentiableFunction.
%
%   solver = ode15sForwardSensitivity(odefun, tspan)
%   solver = ode15sForwardSensitivity(odefun, tspan, odeOpts)
%
%   Then call:
%     [tout, yout] = solver(x0, theta)
%
%   where x0 is dlarray (n-by-1) and theta is dlarray (m-by-1).
%   Returns tout (nSteps-by-1) and yout (nSteps-by-n), matching bdf15s /
%   bdf15 output format.
%
%   odeOpts is an odeset struct. Supported options:
%     RelTol, AbsTol, MaxStep, InitialStep, MaxOrder, BDF
%     Jacobian  - function handle: [dfdy, dfdtheta] = jacfcn(t, y, theta)
%     JPattern  - sparsity pattern of df/dy (n-by-n). Enables sparse
%                 finite-difference Jacobian via column grouping.
%                 Ignored when Jacobian is set.
%   Other odeset fields are ignored.
%
%   The backward pass uses forward sensitivity analysis through the BDF
%   solver to compute exact discrete gradients.

    properties
        OdeFun
        Tspan
        Options
    end

    methods
        function obj = ode15sForwardSensitivity(odefun, tspan, odeOpts)
            arguments
                odefun function_handle
                tspan (1,:)
                odeOpts = []
            end
            % 2 outputs (tout, yout), 2 memory values (Sx0T, SthetaT)
            obj@deep.DifferentiableFunction(2, ...
                NumMemoryValues=2, ...
                SaveInputsForBackward=true, ...
                SaveOutputsForBackward=false);
            obj.OdeFun = odefun;
            obj.Tspan = tspan;
            obj.Options = odeOpts;
        end

        function [tout, yout, Sx0T, SthetaT] = forward(obj, x0, theta)
            x0Val = extractdata(x0);
            if isempty(theta)
                thetaVal = [];
            else
                thetaVal = extractdata(theta);
            end

            odefunRaw = obj.OdeFun;
            tspan = obj.Tspan;
            options = obj.Options;
            t0 = tspan(1);
            tf = tspan(end);

            [tout, yout, Sx0T, SthetaT] = bdf15sForwardSensitivity(...
                odefunRaw, [t0; tf], x0Val, thetaVal, options);
        end

        function [dLdx0, dLdtheta] = backward(~, dLdtout, dLdyout, computeGradients, x0, theta, Sx0T, SthetaT)
            % dLdtout: derivative of loss w.r.t. tout (ignored — time
            %          points are not differentiable w.r.t. x0/theta)
            % dLdyout: derivative of loss w.r.t. yout (nSteps-by-n)
            %
            % Only the final row of dLdyout contributes, since the
            % sensitivity matrices are at the final time.
            dLdyoutVal = extractdata(dLdyout);
            dLdxtVal = dLdyoutVal(end, :);  % 1-by-n

            if computeGradients(1)
                dLdx0 = (dLdxtVal * Sx0T)';
            else
                dLdx0 = zeros(size(extractdata(x0)));
            end

            if computeGradients(2)
                dLdtheta = (dLdxtVal * SthetaT)';
            else
                if isempty(theta)
                    dLdtheta = [];
                else
                    dLdtheta = zeros(size(extractdata(theta)));
                end
            end
        end
    end
end
