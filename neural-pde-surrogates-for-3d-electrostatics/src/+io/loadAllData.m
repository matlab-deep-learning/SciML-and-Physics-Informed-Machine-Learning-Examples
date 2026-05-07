function data = loadAllData(dataRootDir)
    % loadAllData Load all simulation target data from a data directory.
    %
    %   data = io.loadAllData(dataRootDir) loads every .mat file found in
    %   dataRootDir and returns the contents as a struct array. Each .mat
    %   file is expected to contain a variable D, which becomes one element
    %   of the output array.
    %
    %   Input:
    %       dataRootDir - Path to the directory containing .mat data files.
    %
    %   Output:
    %       data - Struct array where each element corresponds to the D
    %              variable from one .mat file in dataRootDir.

    %   Copyright 2026 The MathWorks, Inc.
    metadata = dir(fullfile(dataRootDir,"*.mat"));
    fname = fullfile(dataRootDir,{metadata.name});
    
        function data = loadData(fname)
            data = load(fname);
            data = data.D;
        end
    
    data = arrayfun(@loadData, fname, UniformOutput = true);
end