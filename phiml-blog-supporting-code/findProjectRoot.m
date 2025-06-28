function rootDir = findProjectRoot(marker)
%FINDPROJECTROOT Locate the root directory of the project.
%   rootDir = FINDPROJECTROOT(marker) searches upward from the current
%   directory to find a directory containing the file or folder specified
%   by 'marker' (e.g., 'startup.m', '.git', 'generatePendulumData.m').
%   Returns the path to that directory.
%
%   Example:
%       rootDir = findProjectRoot('startup.m');

if nargin < 1
    marker = 'startup.m'; % Default marker file
end

currentDir = pwd;
while true
    if exist(fullfile(currentDir, marker), 'file') || ...
       exist(fullfile(currentDir, marker), 'dir')
        rootDir = currentDir;
        return;
    end
    [parentDir, ~, ~] = fileparts(currentDir);
    if strcmp(currentDir, parentDir)
        error('Project root not found. Marker "%s" not found in any parent directory.', marker);
    end
    currentDir = parentDir;
end
end