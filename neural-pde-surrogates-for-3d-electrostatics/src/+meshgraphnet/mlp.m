function x = mlp(params,x,numLayers)
    % mlp Forward pass of a multi-layer perceptron with residual connections.
    %
    %   x = meshgraphnet.mlp(params, x, numLayers) applies numLayers fully
    %   connected layers with GeLU activations and residual connections.
    %   Residual connections are added from the second layer onward, where
    %   layer sizes match (see initializeMLP).
    %
    %   Inputs:
    %       params    - Struct of MLP parameters with fields fc_1, ...,
    %                   fc_numLayers, each containing W and b.
    %       x         - Input features of size D-by-N or D-by-N-by-B
    %                   (batched via pagemtimes).
    %       numLayers - Number of layers in the MLP.
    %
    %   Output:
    %       x - Output features.

    %   Copyright 2026 The MathWorks, Inc.
    
    for i = 1:(numLayers-1)
        p = params.("fc_"+i);
        z = pagemtimes(p.W,x) + p.b;
        z = gelu(z);
        if i > 1
            % Add a residual connection
            x = x + z;
        else
            x = z;
        end
    end
    p = params.("fc_"+numLayers);
    z = pagemtimes(p.W,x) + p.b;
    if size(x,1) == size(z,1)
        x = x + z;
    else
        x = z;
    end
end