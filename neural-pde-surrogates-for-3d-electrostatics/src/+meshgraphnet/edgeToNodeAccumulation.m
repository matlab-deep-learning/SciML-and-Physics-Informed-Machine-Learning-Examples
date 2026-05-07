function x = edgeToNodeAccumulation(x,xe,B)
    % edgeToNodeAccumulation Accumulate edge features onto nodes and add to node features.
    %
    %   x = meshgraphnet.edgeToNodeAccumulation(x, xe, B) accumulates the
    %   edge features xe onto target nodes using the node-edge adjacency
    %   matrix B and adds the result to the node features x.
    %
    %   Inputs:
    %       x  - Node feature matrix of size H-by-NumNodes, where H is the
    %            hidden feature dimension.
    %       xe - Edge feature matrix of size H-by-NumEdges. The hidden
    %            dimension H must match that of x.
    %       B  - Node-edge adjacency matrix of size NumNodes-by-NumEdges.
    %
    %   Output:
    %       x  - Updated node feature matrix of size H-by-NumNodes, computed
    %            as x + xe * B'.

    %   Copyright 2026 The MathWorks, Inc.
    xe = xe*B';
    x = x+xe;
end 