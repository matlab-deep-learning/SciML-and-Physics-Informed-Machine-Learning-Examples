function downloadSimulationData(url,destination)
% The downloadSimulationData function downloads pregenerated simulation
% data for the 3D battery heat analysis problem.

% Copyright 2026 The MathWorks, Inc.

    if ~exist(destination,"dir")
        mkdir(destination);
    end
    
    [~,name,filetype] = fileparts(url);
    netFileFullPath = fullfile(destination,name+filetype);
    
    % Check for the existence of the file and download the file if it does not
    % exist
    if ~exist(netFileFullPath,"file")
        disp("Downloading simulation data.");
        disp("This can take several minutes to download...");
        websave(netFileFullPath,url);
    
        % If the file is a ZIP file, extract it
        if filetype == ".zip"
            unzip(netFileFullPath,destination)
        end
        disp("Done.");
    
    end
end