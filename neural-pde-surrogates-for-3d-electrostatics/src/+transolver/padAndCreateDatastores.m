function [cds,cdsVal] = padAndCreateDatastores(...
    normalizedInputsTrain,normalizedInputsVal,...
    normalizedTargetsTrain,normalizedTargetsVal)
    % padAndCreateDatastores Pad variable-length sequences and create combined datastores.
    %
    %   [cds, cdsVal] = transolver.padAndCreateDatastores(
    %   normalizedInputsTrain, normalizedInputsVal,
    %   normalizedTargetsTrain, normalizedTargetsVal) pads the input and
    %   target cell arrays to uniform length, creates array datastores for
    %   each component, and combines them. Training inputs use a
    %   RandomAugmentationDatastore for data augmentation.
    %
    %   Inputs:
    %       normalizedInputsTrain  - Cell array of normalized training inputs.
    %       normalizedInputsVal    - Cell array of normalized validation inputs.
    %       normalizedTargetsTrain - Cell array of normalized training targets.
    %       normalizedTargetsVal   - Cell array of normalized validation targets.
    %
    %   Outputs:
    %       cds    - Combined training datastore (inputs, mask, targets, mask).
    %       cdsVal - Combined validation datastore (inputs, mask, targets, mask).

    %   Copyright 2026 The MathWorks, Inc.

    [paddedInputsTrain,maskTrain] = padsequences(normalizedInputsTrain,2);
    [paddedInputsVal,maskVal] = padsequences(normalizedInputsVal,2);
    paddedTargetsTrain = padsequences(normalizedTargetsTrain,2);
    paddedTargetsVal = padsequences(normalizedTargetsVal,2);
    
    inputDsTrain = transolver.datastore.RandomAugmentationDatastore(paddedInputsTrain);
    maskDsTrain = arrayDatastore(maskTrain(1,:,:),IterationDimension=3);
    targetDsTrain = arrayDatastore(paddedTargetsTrain,IterationDimension = 3);
    inputDsVal = arrayDatastore(paddedInputsVal,IterationDimension=3);
    maskDsVal = arrayDatastore(maskVal(1,:,:),IterationDimension=3);
    targetDsVal = arrayDatastore(paddedTargetsVal,IterationDimension=3);
    
    cds = combine(inputDsTrain,maskDsTrain,targetDsTrain,maskDsTrain);
    cdsVal = combine(inputDsVal,maskDsVal,targetDsVal,maskDsVal);
end