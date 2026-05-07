function [X,Xe,B,Y] = createMiniBatch(X,Xe,B,Y,useGpu)
    % createMiniBatch Assemble a mini-batch from cells of graph data.
    %
    %   [X,Xe,B,Y] = meshgraphnet.createMiniBatch(X,Xe,B,Y,useGpu)
    %   concatenates cell arrays of per-sample node features, edge features,
    %   node-edge adjacency matrices, and targets into batched arrays. The
    %   adjacency matrices are combined via blkdiag and row-normalized.
    %
    %   Inputs:
    %       X      - Cell array of node feature matrices (each H-by-Ni).
    %       Xe     - Cell array of edge feature matrices (each H-by-Ei).
    %       B      - Cell array of sparse node-edge adjacency matrices.
    %       Y      - Cell array of target matrices (each D-by-Ni).
    %       useGpu - Logical flag to move data to the GPU.
    %
    %   Outputs:
    %       X  - Batched dlarray of node features.
    %       Xe - Batched dlarray of edge features.
    %       B  - Row-normalized block-diagonal adjacency matrix.
    %       Y  - Batched dlarray of targets.
    
    %   Copyright 2026 The MathWorks, Inc.
    if useGpu
        X = gpuArray(dlarray(cat(2,X{:})));
    else
        X = dlarray(cat(2,X{:}));
    end
    B = blkdiag(B{:});

    
    % Normalize by num edges connected to a node. 
    B = B ./ sum(B,2);
    if useGpu
        Xe = gpuArray(dlarray(cat(2,Xe{:})));
        Y = gpuArray(dlarray(cat(2,Y{:})));
        B = gpuArray(B);
    else
        Xe = dlarray(cat(2,Xe{:}));
        Y = dlarray(cat(2,Y{:}));
    end
end