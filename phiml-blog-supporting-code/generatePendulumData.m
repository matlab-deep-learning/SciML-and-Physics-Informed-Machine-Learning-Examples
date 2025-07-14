function data = generatePendulumData(omega0,x0,tSpan,noiseLevel,doPlot)
%Generate the noisy pendulum trajectory data to be used in the examples
%   NeuralODE_nonlinear_pendulum
%   HNN_nonlinear_pendulum
%   PINN_nonlinear_pendulum
%The output dataT is a table with the following variables:
%   t: the times when the quantities are observed
%   thetaNoisy: noisy measurements of the angular position
%   omegaNoisy: noisy measurements of the angular velocity
%   thetaDot: numerical derivative of thetaNoisy (after smoothing)
%   omegaDot: numerical derivative of omegaNoisy (after smoothing)

rng(0); % for reproducibility

F = ode;
F.ODEFcn = @(t,x) [x(2); -omega0^2.*sin(x(1))];
F.InitialValue = x0;
F.Solver = "ode45";
sol = solve(F,tSpan);

% Add noise to the data measurements of theta and omega
yOut = sol.Solution;
tOut = sol.Time;
theta = yOut(1,:);
omega = yOut(2,:);
thetaNoisy = theta + noiseLevel*randn(size(theta));
omegaNoisy = omega + noiseLevel*randn(size(omega));

% Smooth before numerically approximating derivatives
thetaSmooth = sgolayfilt(thetaNoisy, 3, 7);
omegaSmooth = sgolayfilt(omegaNoisy, 3, 7);

% Compute derivatives 
thetaDot = gradient(thetaSmooth, tOut);
omegaDot = gradient(omegaSmooth, tOut);

% Plotting (Optional)
if doPlot
    % Plot phase portrait (q, p)
    figure;
    plot(theta, omega, 'b-', 'LineWidth', 3); hold on;
    plot(thetaNoisy, omegaNoisy, 'ro', 'MarkerSize', 6); hold off
    xlabel('\theta (rad)');
    ylabel('\omega (rad/s)');
    legend('True trajectory', 'Noisy data');
    title('Pendulum Phase Portrait');
    ax = gca;
    ax.FontSize = 16;
    ax.LineWidth = 1.5;

    % Plot time series
    figure;
    subplot(2,1,1);
    plot(tOut, theta, 'b-','LineWidth',3); hold on
    plot(tOut, thetaNoisy, 'ro');
    xlabel('Time (s)'); ylabel('\theta (rad)');
    legend('True', 'Data');
    title('Angle vs Time');
    % xlim([0 4]);
    ax = gca;
    ax.FontSize = 16;
    ax.LineWidth = 1.5;
    hold off
    
    subplot(2,1,2);
    plot(tOut, omega, 'b-', 'LineWidth',3); hold on
    plot(tOut, omegaNoisy, 'ro');
    xlabel('Time (s)'); ylabel('\omega (rad/s)');
    legend('True', 'Data');
    title('Angular Velocity vs Time');
    % xlim([0 4]);
    ax = gca;
    ax.FontSize = 16;
    ax.LineWidth = 1.5;
    hold off
end

t = tOut;

% save column vectors to data struct
data.t = t'; 
data.thetaNoisy = thetaNoisy';
data.omegaNoisy = omegaNoisy';
data.thetaDot = thetaDot';
data.omegaDot = omegaDot';

% Find the project root 
rootDir = findProjectRoot('startup.m'); 
dataDir = fullfile(rootDir, 'pendulumData');

% Create directory if it doesn't exist
if ~exist(dataDir, 'dir')
    mkdir(dataDir);
end

% Save the data
save(fullfile(dataDir, 'pendulum_qp_dqdp.mat'), 'data');

fprintf('Pendulum data written to pendulum_qp_dqdp.mat \n');


end