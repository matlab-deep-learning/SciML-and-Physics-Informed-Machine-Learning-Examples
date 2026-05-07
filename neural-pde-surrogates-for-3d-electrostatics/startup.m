% Get the project root folder
projectRoot = fileparts(mfilename('fullpath'));

% Add the project root to the path
addpath(projectRoot);

% Add all subfolders to the path, create them if they don't exist
dirs = {'data','STL','results','src'};
for k=1:numel(dirs)
    d = fullfile(projectRoot, dirs{k});
    if ~exist(d, 'dir')
        mkdir(d);
    end
    addpath(d);
end

% Display a message
disp(['Project path set. Root: ', projectRoot]);