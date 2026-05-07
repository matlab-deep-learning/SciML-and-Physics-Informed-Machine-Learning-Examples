function xe = nodeToEdgeAccumulation(x,xe,B)
    % nodeToEdgeAccumulation Accumulate node features onto edges and add to edge features.
    %
    %   xe = meshgraphnet.nodeToEdgeAccumulation(x, xe, B) accumulates the
    %   node features x onto edges using the node-edge adjacency matrix B
    %   and adds the result to the edge features xe.
    %
    %   Inputs:
    %       x  - Node feature matrix of size H-by-NumNodes, where H is the
    %            hidden feature dimension.
    %       xe - Edge feature matrix of size H-by-NumEdges. The hidden
    %            dimension H must match that of x.
    %       B  - Node-edge adjacency matrix of size NumNodes-by-NumEdges.
    %
    %   Output:
    %       xe - Updated edge feature matrix of size H-by-NumEdges, computed
    %            as x * B + xe.

    %   Copyright 2026 The MathWorks, Inc.
    x = x*B;
    xe = x + xe;
end