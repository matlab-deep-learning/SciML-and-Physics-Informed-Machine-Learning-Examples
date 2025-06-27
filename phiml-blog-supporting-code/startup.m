% Get the project root folder
projectRoot = fileparts(mfilename('fullpath'));

% Add the project root to the path
addpath(projectRoot);

% Add all subfolders to the path
addpath(genpath(projectRoot));

% Display a message
disp(['Project path set. Root: ', projectRoot]);