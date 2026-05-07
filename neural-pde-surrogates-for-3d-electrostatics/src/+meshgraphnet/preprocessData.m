function [X,Xe,Y,A,B] = preprocessData(data,keep)
    % preprocessData Extract graph features and targets from simulation data.
    %
    %   [X,Xe,Y,A,B] = meshgraphnet.preprocessData(data, keep) extracts
    %   node features (coordinates), edge features (displacements and
    %   distances), targets (electric potential), the node adjacency matrix,
    %   and the node-edge adjacency matrix from a simulation data struct.
    %
    %   Inputs:
    %       data - Struct with fields Mesh (FEMesh) and optionally
    %              Potential (nodal potential values).
    %       keep - Indices of nodes to retain. Use [] to keep all nodes.
    %
    %   Outputs:
    %       X  - Node feature matrix (3-by-NumNodes, coordinates).
    %       Xe - Edge feature matrix (4-by-NumEdges, displacements and
    %            Euclidean distance).
    %       Y  - Target matrix (1-by-NumNodes, potential), or [] if
    %            Potential is not present in data, as in inference.
    %       A  - Sparse symmetric node adjacency matrix.
    %       B  - Sparse node-edge adjacency matrix.

    %   Copyright 2026 The MathWorks, Inc.

    % Only consider the nodes designated by "keep"
    if isempty(keep)
        keep = 1:size(data.Mesh.Nodes,2);
    end
    X = data.Mesh.Nodes(:,keep);
    if isfield(data,'Potential')
        Y = data.Potential(keep)';
    else
        Y = []; % don't return targets
    end
    
    % Get the adjacency matrix
    A = meshgraphnet.mesh2adjacency(data.Mesh);
    
    % Only retain the nodes designated as keep.
    A = A(keep,:);
    A = A(:,keep);
    
    % Construct the node-edge adjacency matrix, a matrix B of size 
    % NumNodes x NumEdges
    % where B(i,j) = 1 if node i is one of the two nodes attached to edge j.
    % This will be used in MGN to accumulate edge
    % features onto nodes and node features onto edges.
    % Recompute R,C here to include the self loops and the edges from 
    % symmetrisation above.
    [r,c] = find(A);
    rc = [r;c];
    
    % Number the edges 1 -> numEdges following R and C
    edgeLabels = [(1:numel(r))';(1:numel(r))'];
    
    % Construct the sparse node-edge adjacency.
    B = sparse(rc,edgeLabels,ones(size(rc)),size(A,1),numel(r));
    B = double(B>0);
    
    % Construct edge features. Here use displacements and lengths.
    Xe = X(:,r) - X(:,c);
    Xe = [Xe; sqrt(sum(Xe(1:3,:).^2,1))];
end