function p = initializeMLP(inputSize,hiddenSize,numLayers,outputSize)
    % initializeMLP Initialize learnable parameters for an MLP.
    %
    %   p = meshgraphnet.initializeMLP(inputSize, hiddenSize, numLayers)
    %   creates a struct of weight matrices and bias vectors for an MLP with
    %   the specified number of layers. Layer sizes are
    %   [inputSize, hiddenSize, ..., hiddenSize]. Weights are initialized
    %   using Glorot uniform initialization and biases are initialized to
    %   zero.
    %
    %   p = meshgraphnet.initializeMLP(inputSize, hiddenSize, numLayers,
    %   outputSize) uses outputSize for the final layer instead of
    %   hiddenSize, giving layer sizes [inputSize, hiddenSize, ...,
    %   hiddenSize, outputSize].
    %
    %   Inputs:
    %       inputSize  - Size of the input features.
    %       hiddenSize - Size of each hidden layer.
    %       numLayers  - Number of layers in the MLP.
    %       outputSize - Size of the final layer output. Default: hiddenSize.
    %
    %   Output:
    %       p - Struct with fields fc_1, fc_2, ..., fc_numLayers, each
    %           containing W (weight matrix) and b (bias vector).

    %   Copyright 2026 The MathWorks, Inc.
    if nargin < 4
        outputSize = hiddenSize;
    end
    p = struct();
    sz = [inputSize, repelem(hiddenSize,numLayers-1), outputSize];
    for i = 1:numLayers
        p.("fc_"+i).W = meshgraphnet.initializeGlorot(sz(i),sz(i+1));
        p.("fc_"+i).b = zeros(sz(i+1),1);
    end
end
