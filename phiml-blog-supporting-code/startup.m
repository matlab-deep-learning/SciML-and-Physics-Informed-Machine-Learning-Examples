% Get the project root folder
projectRoot = fileparts(mfilename('fullpath'));

% Add the project root to the path
addpath(projectRoot);

% Add all subfolders to the path
addpath(fullfile(projectRoot, 'HNN'));
addpath(fullfile(projectRoot, 'PINN'));
addpath(fullfile(projectRoot, 'NeuralODE'));
addpath(fullfile(projectRoot, 'FNO'));
addpath(fullfile(projectRoot, 'UDE_SINDy'));

% Display a message
disp(['Project path set. Root: ', projectRoot]);