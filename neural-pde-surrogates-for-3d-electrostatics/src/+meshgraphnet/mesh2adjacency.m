function A = mesh2adjacency(mesh)
    % mesh2adjacency Build a symmetric node adjacency matrix from a 3-D quadratic FEA mesh.
    %
    %   A = meshgraphnet.mesh2adjacency(mesh) constructs a sparse symmetric
    %   adjacency matrix from a quadratic-order FEMesh. Self-loops are
    %   included, and all entries are 0 or 1.
    %
    %   Input:
    %       mesh - An FEMesh object for 3-D quadratic-order elements.
    %
    %   Output:
    %       A - Sparse symmetric adjacency matrix of size NumNodes-by-NumNodes.
    
    %   Copyright 2026 The MathWorks, Inc.
    numNodes = size(mesh.Nodes,2);
    
    % https://www.mathworks.com/help/pde/ug/mesh-data.html
    % Specify the edges from the above webpage.
    r = [1,5,2,6,3,10,4,8,1,7,2,9];
    c = [5,2,6,3,10,4,8,1,7,3,9,4];
    
    % Get the node indices of those edges from the elements.
    R = mesh.Elements(r,:);
    C = mesh.Elements(c,:);
    
    % Construct a sparse adjacency matrix.
    V = ones(size(R));
    R = reshape(R,[],1);
    C = reshape(C,[],1);
    V = reshape(V,[],1);
    A = sparse(R,C,V,numNodes,numNodes);
    
    % Add self loops 
    A = A + speye(numNodes);
    
    % Symmetrise the adjacency matrix
    A = double((A>0) | (A'>0));
end