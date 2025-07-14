function [fSamples, thetaSamples, grid] = fnoDataHelper(numModes, numSamples, omega0, x0, gridSize, doPlot)
    % Generate samples of the forcing function f from N(0, (-Δ + I)^(-1))
    % numModes: number of Fourier modes
    % numSamples: number of samples to generate
    % omega0: natural frequency
    % x0: initial condition
    % gridSize: number of points in t domain
    % doPlot: option to plot samples

    % Example usage
    % fSamples = fnoDataHelper(5,2000,1,[0;1],512,0);
    
    rng(0); % for reproducibility

    % Define the x domain
    xMax = 10;
    tSol = linspace(0, xMax, gridSize);

    % Initialize storage for samples
    fSamples = zeros(numSamples, length(tSol));
    thetaSamples = zeros(numSamples, length(tSol));

    % scaling factor
    sigma = 0.5;
    
    % Generate samples
    for j = 1:numSamples
        % Initialize the sample
        f = zeros(size(tSol));

        % Generate the Fourier series
        for k = 1:numModes
            % Compute the eigenvalue for the covariance operator
            lambda_k = 1 / (k^2 + 1);

            % Generate random coefficients
            a_k = sigma*sqrt(lambda_k) * randn;
            b_k = sigma*sqrt(lambda_k) * randn;

            % Update the sample with the cosine and sine terms
            f = f + a_k * cos(2 * pi * k * tSol/xMax) + b_k * sin(2 * pi * k * tSol/xMax);
        end

        % Store the sample
        fSamples(j, :) = f;

        % Solve the ode for theta
        F = ode;
        fInterp = @(x) interp1(tSol, f, x, 'spline', 'extrap');
        F.ODEFcn = @(t, y) [y(2); -omega0^2 * sin(y(1)) + fInterp(t)];
        F.InitialValue = x0;
        F.Solver = "ode45";
        sol = solve(F,tSol);
        thetaSamples(j, :) = sol.Solution(1,:);

    end
    
    if doPlot
        % Plot a few samples
        figure;
        hold on;
        for i = 1:min(numSamples, 5)
            plot(tSol, fSamples(i, :));
        end
        title('Samples of Forcing Function f');
        xlabel('x');
        ylabel('f(x)');
        hold off;
    
        figure;
        hold on;
        for i = 1:min(numSamples, 5)
            plot(tSol, thetaSamples(i, :));
        end
        title('Samples of Solution Function \theta');
        xlabel('x');
        ylabel('f(x)');
        hold off;
    end
    
    grid = tSol;
end

