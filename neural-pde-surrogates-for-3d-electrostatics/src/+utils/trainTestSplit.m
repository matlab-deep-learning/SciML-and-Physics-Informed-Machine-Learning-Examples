function [trainIdx,valIdx] = trainTestSplit(numObs,trainRatio)
    % trainTestSplit Randomly split observation indices into training and validation sets.
    %
    %   [trainIdx, valIdx] = utils.trainTestSplit(numObs, trainRatio)
    %   returns randomly permuted indices split according to trainRatio.
    %
    %   Inputs:
    %       numObs     - Total number of observations.
    %       trainRatio - Fraction of observations for training (0 to 1).
    %
    %   Outputs:
    %       trainIdx - Indices for the training set.
    %       valIdx   - Indices for the validation set.

    %   Copyright 2026 The MathWorks, Inc.

    idx = randperm(numObs);
    numTrain = floor(trainRatio*numObs);
    trainIdx = idx(1:numTrain);
    valIdx = idx(numTrain+1:end);
end