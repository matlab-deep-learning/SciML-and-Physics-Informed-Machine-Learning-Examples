function data = generatePendulumDataWithDamping(omega0,x0,tSpan,noiseLevel,doPlot)
%Generate the noisy pendulum trajectory data to be used in the examples
%   UDE_nonlinear_pendulum_damping
%The output dataT is a table with the following variables:
%   t: the times when the quantities are observed
%   thetaNoisy: noisy measurements of the angular position
%   omegaNoisy: noisy measurements of the angular velocity
%   thetaDot: numerical derivative of thetaNoisy (after smoothing)
%   omegaDot: numerical derivative of omegaNoisy (after smoothing)
%   F_data: measured damping force f
%   F_true: true damping force f

rng(0); % for reproducibility

c1 = 0.2; % linear friction coefficient
c2 = 0.1; % nonlinear friction coefficient

F = ode;
F.ODEFcn = @(t,x) [x(2); -omega0^2.*sin(x(1)) - (c1 + c2*x(2))*x(2)];
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

% Compute true friction force (for reference)
F_true = -(c1 + c2*omega).*omega;

% Smooth noisy data before approximating derivatives
thetaSmooth = sgolayfilt(thetaNoisy, 3, 7);
omegaSmooth = sgolayfilt(omegaNoisy, 3, 7);

% Numerically approximate derivatives
thetaDot = gradient(thetaSmooth, tOut);
omegaDot = gradient(omegaSmooth, tOut);

% Compute the "residual" friction force from the data
% Rearranged ODE: F = omegaDot + omega_0^2*sin(theta)
F_data = omegaDot + omega0^2 * sin(thetaSmooth);

if doPlot
    % Plot phase portrait
    figure;
    plot(theta, omega, 'b-', 'LineWidth', 3); hold on;
    plot(thetaNoisy, omegaNoisy, 'ro', 'MarkerSize', 6);
    xlabel('\theta (rad)');
    ylabel('\omega (rad/s)');
    legend('True trajectory', 'Noisy data');
    title('Pendulum Phase Portrait');
    ax = gca;
    ax.FontSize = 16;
    ax.LineWidth = 1.5;

    % Plot time series
    figure;
    subplot(3,1,1);
    plot(tOut, theta, 'b-','LineWidth',3); hold on
    plot(tOut, thetaNoisy, 'ro');
    xlabel('Time (s)'); ylabel('\theta (rad)');
    legend('True', 'Data');
    title('Angle vs Time');
    ax = gca; ax.FontSize = 16; ax.LineWidth = 1.5; hold off
    
    subplot(3,1,2);
    plot(tOut, omega, 'b-', 'LineWidth',3); hold on
    plot(tOut, omegaNoisy, 'ro');
    xlabel('Time (s)'); ylabel('\omega (rad/s)');
    legend('True', 'Data');
    title('Angular Velocity vs Time');
    ax = gca; ax.FontSize = 16; ax.LineWidth = 1.5; hold off
    
    subplot(3,1,3);
    plot(tOut, F_true, 'k-', 'LineWidth',2); hold on
    plot(tOut, F_data, 'r--', 'LineWidth',2);
    xlabel('Time (s)'); ylabel('Friction F');
    legend('True F', 'Estimated F (from data)');
    title('Friction Force');
    ax = gca; ax.FontSize = 16; ax.LineWidth = 1.5; hold off
end

t = tOut;

% save column vectors to data struct
data.t = t'; 
data.thetaNoisy = thetaNoisy';
data.omegaNoisy = omegaNoisy';
data.thetaDot = thetaDot';
data.omegaDot = omegaDot';
data.F_data = F_data';
data.F_true = F_true';

% Find the project root 
rootDir = findProjectRoot('startup.m'); 
dataDir = fullfile(rootDir, 'pendulumData');

% Create directory if it doesn't exist
if ~exist(dataDir, 'dir')
    mkdir(dataDir);
end

% Save the data
save(fullfile(dataDir, 'pendulum_with_damping_qp_dqdp_F.mat'), 'data');

disp('Pendulum data written to pendulum_with_damping_qp_dqdp_F.mat \n');


end