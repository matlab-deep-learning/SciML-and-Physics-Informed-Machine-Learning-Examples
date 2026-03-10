function [sequenceData, sampleWeights] = prepareSequenceData(data, profilesList, profileSizes, inputCols, targetCols)
% PREPARESEQUENCEDATA Transforms tabular time-series data into a 3-D dlarray format
% suitable for training a recurrent neural network.
%
% This function:
% 1. Aligns inputs at time t with targets at t+1 to enable next-step forecasting
% 2. Creates a 3-D dlarray object (C×B×T format): features × profiles × time steps
% 3. Generates binary sample weights to mask padded data during training
% 4. Handles variable-length sequences with zero-padding and converts to single precision
%
% Inputs:
%   data - Table containing the raw sensor data
%   profilesList - List of profile IDs to include
%   profileSizes - Table with profile_id and corresponding sequence lengths
%   inputCols - Cell array of column names to use as inputs
%   targetCols - Cell array of column names to use as targets
%
% Outputs:
%   sequenceData - 3-D dlarray object with dimensions [numFeatures+numTargets, numProfiles, maxTimeSteps-1]
%   sampleWeights - Binary mask with dimensions [1, numProfiles, maxTimeSteps-1]

%   Copyright 2025 The MathWorks, Inc.

% Get the maximum profile length
numInputs = numel(inputCols);
numTargets = numel(targetCols);
max_profile_length = max(profileSizes.GroupCount(ismember(profileSizes.profile_id, profilesList)));

% Initialize dlarray with NaNs
sequenceData = dlarray(nan(numInputs+numTargets,length(profilesList),max_profile_length-1),'CBT');

% Fill the dlarray object with data
for i = 1:length(profilesList)
    pid = profilesList(i);
    df = data(data.profile_id == pid, :);
    % Convert table to array and drop 'profile_id' column.  Apply time step
    % shift for output columns: inputs at time step t are used to predict
    % the outputs at timestep t+1
    sequenceData(1:numInputs, i, 1:height(df)-1) = table2array(df(1:end-1, inputCols))';
    sequenceData(numInputs+1:end, i, 1:height(df)-1) = table2array(df(2:end, targetCols))';
end

% Create sample weights (binary mask for valid data points)
sampleWeights = ~isnan(sequenceData(1, :, :));

% Replace NaNs with zeros
sequenceData(isnan(sequenceData)) = 0;

% Convert to single precision
sequenceData = single(sequenceData);
end