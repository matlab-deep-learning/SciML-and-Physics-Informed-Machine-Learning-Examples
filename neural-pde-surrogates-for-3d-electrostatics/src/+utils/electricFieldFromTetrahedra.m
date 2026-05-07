function out = electricFieldFromTetrahedra(nodes, elements, V)
    %electricFieldFromTetrahedra Compute electric field from nodal potentials on tetrahedra.
    %   out = electricFieldFromTetrahedra(nodes, elements, V) computes E = -grad(V)
    %   on Tet4 or Tet10 meshes from PDE Toolbox.
    %   For Tet4: grad(V) is constant per element.
    %   For Tet10: grad(V) is linear per element, evaluated at quadrature
    %   points (for element values) and at each node's barycentric position
    %   (for nodal values).
    %
    %   Inputs:
    %       nodes    - Node coordinates (3xN or Nx3)
    %       elements - Tetrahedral connectivity (4xNe/10xNe or Nex4/Nex10)
    %       V        - Nodal electric potential (Nx1 or 1xN)
    %
    %   Outputs:
    %       out.E_node    - Nx3 nodal electric field [Ex Ey Ez]
    %       out.E_elem    - Nex3 element electric field (quadrature average
    %                       for Tet10, constant for Tet4)
    %       out.Emag_node - Nx1 nodal electric field magnitude
    %       out.Emag_elem - Nex1 element electric field magnitude
    
    %   Copyright 2026 The MathWorks, Inc.
  
    % Normalize inputs
      if size(nodes,1)==3 && size(nodes,2)~=3,  X = nodes.'; else, X = nodes;  end
      V = V(:);   % ensure column vector
    
      % Normalize elements to Ne x nen
      [r,c] = size(elements);
      if     r==4 || r==10,  T = elements.';
      elseif c==4 || c==10,  T = elements;
      else,  error('elements must be 4- or 10-node tetrahedral connectivity.');
      end
      [Ne, nen] = size(T);
      if nen~=4 && nen~=10, error('Only Tet4 and Tet10 supported.'); end
    
      N = size(X,1);
    
      E_node_sum = zeros(N, 3);
      E_node_cnt = zeros(N, 1);
    
      if nen == 4
          % Tet4 version
          ids = T';   % 4 x Ne
    
          % Build [1 x y z] matrices for all elements: 4 x 4 x Ne
          Xe_all = reshape(X(ids(:),:), 4, Ne, 3);
          Xe_all = permute(Xe_all, [1,3,2]);           % 4 x 3 x Ne
          A_all = cat(2, ones(4,1,Ne), Xe_all);         % 4 x 4 x Ne
    
          % Batch solve A \ I for all elements
          I4 = repmat(eye(4), 1, 1, Ne);
          C_all = pagemldivide(A_all, I4);              % 4 x 4 x Ne
          dN_all = C_all(2:4,:,:);                      % 3 x 4 x Ne
    
          % Potentials at element nodes: 4 x 1 x Ne
          Ve_all = reshape(V(ids(:)), 4, Ne);
          Ve_all = reshape(Ve_all, 4, 1, Ne);
    
          % Gradient of V for all elements: 3 x 1 x Ne
          gradV_all = pagemtimes(dN_all, Ve_all);
          E_elem = -squeeze(gradV_all)';                % Ne x 3
    
          % Accumulate element E-field to nodes
          for d = 1:3
              vals = repmat(E_elem(:,d)', 4, 1);       % 4 x Ne
              E_node_sum(:,d) = accumarray(ids(:), vals(:), [N,1]);
          end
          E_node_cnt = accumarray(ids(:), 1, [N,1]);
    
      else
          % Tet10 version
          % Nodal barycentric coordinates in SF canonical order
          nodeRST = [
              1.0  0.0  0.0;   % N1  (corner)
              0.0  1.0  0.0;   % N2  (corner)
              0.0  0.0  1.0;   % N3  (corner)
              0.0  0.0  0.0;   % N4  (corner, L4=1)
              0.5  0.5  0.0;   % N5:  edge 1-2
              0.0  0.5  0.5;   % N6:  edge 2-3
              0.5  0.0  0.5;   % N7:  edge 3-1
              0.5  0.0  0.0;   % N8:  edge 1-4
              0.0  0.5  0.0;   % N9:  edge 2-4
              0.0  0.0  0.5];  % N10: edge 3-4
    
          % 4-point quadrature for element-average
          a = 0.5854101966249685; b = 0.1381966011250105;
          quadPts = [a b b; b a b; b b a; b b b];
    
          % PDE Toolbox convention: nodes 1-4 are corners, 5-10 are mid-edge.
          % Reorder mid-edge nodes to match canonical SF edge ordering.
          cornerIdx = T(:,1:4);    % Ne x 4
          midIdx    = T(:,5:10);   % Ne x 6
    
          % Canonical edges connecting corners
          edgePairs = [1 2; 2 3; 3 1; 1 4; 2 4; 3 4];
    
          % Compute expected midpoints of canonical edges for all elements
          cornerCoords = reshape(X(cornerIdx(:),:), Ne, 4, 3);   % Ne x 4 x 3
          targets = zeros(Ne, 6, 3);
          for k = 1:6
              targets(:,k,:) = 0.5*(cornerCoords(:,edgePairs(k,1),:) + ...
                                     cornerCoords(:,edgePairs(k,2),:));
          end
    
          % Actual mid-edge node coordinates
          midCoords = reshape(X(midIdx(:),:), Ne, 6, 3);         % Ne x 6 x 3
    
          % Pairwise squared distances between mid-nodes and target midpoints
          dist2 = sum((reshape(midCoords,Ne,6,1,3) - ...
                       reshape(targets,Ne,1,6,3)).^2, 4);         % Ne x 6 x 6
    
          % Match each canonical target to the closest mid-edge node
          [~, assignment] = min(dist2, [], 2);                    % Ne x 1 x 6
          assignment = squeeze(assignment);                        % Ne x 6
          if Ne == 1, assignment = assignment'; end                % handle single-element case
    
          % Reorder mid-edge nodes to canonical SF order
          rowIdx = repmat((1:Ne)', 1, 6);
          reorderedMid = midIdx(sub2ind([Ne,6], rowIdx, assignment));
          ids_sf_all = [cornerIdx, reorderedMid];                 % Ne x 10
    
          % Gather coordinates and potentials for all elements
          Xe_all = reshape(X(ids_sf_all(:),:), Ne, 10, 3);
          Xe_all = permute(Xe_all, [2,3,1]);                      % 10 x 3 x Ne
    
          Ve_all = reshape(V(ids_sf_all(:)), Ne, 10);
          Ve_all = reshape(Ve_all', 10, 1, Ne);                   % 10 x 1 x Ne
    
          % Precompute shape function derivatives
          dN_quad  = zeros(10, 3, 4);
          for q = 1:4
              [dNr,dNs,dNt] = tet10_shape_derivs( ...
                  quadPts(q,1), quadPts(q,2), quadPts(q,3));
              dN_quad(:,:,q) = [dNr dNs dNt];
          end
    
          dN_nodal = zeros(10, 3, 10);
          for n = 1:10
              [dNr,dNs,dNt] = tet10_shape_derivs( ...
                  nodeRST(n,1), nodeRST(n,2), nodeRST(n,3));
              dN_nodal(:,:,n) = [dNr dNs dNt];
          end
    
          % Element-average E-field (4-point quadrature)
          E_elem_acc = zeros(3, 1, Ne);
          for q = 1:4
              dN = dN_quad(:,:,q); % 10 x 3 (fixed)
              % Jacobian: J = Xe' * dN  ->  3 x 3 x Ne
              J = pagemtimes(permute(Xe_all,[2,1,3]), dN);
              % Physical gradients: J' \ dN'  ->  3 x 10 x Ne
              dN_xyz = pagemldivide(permute(J,[2,1,3]), repmat(dN',1,1,Ne));
              % grad(V) = dN_xyz * Ve  ->  3 x 1 x Ne
              E_elem_acc = E_elem_acc - pagemtimes(dN_xyz, Ve_all);
          end
          E_elem = squeeze(E_elem_acc / 4)'; % Ne x 3
          if Ne == 1, E_elem = E_elem(:)'; end
    
          % Nodal E-field (accumulate from element contributions)
          for n = 1:10
              dN = dN_nodal(:,:,n); % 10 x 3 (fixed)
              J = pagemtimes(permute(Xe_all,[2,1,3]), dN);
              dN_xyz = pagemldivide(permute(J,[2,1,3]), repmat(dN',1,1,Ne));
              gradV = pagemtimes(dN_xyz, Ve_all); % 3 x 1 x Ne
              Enode = -squeeze(gradV)'; % Ne x 3
              if Ne == 1, Enode = Enode(:)'; end
    
              nids = ids_sf_all(:, n);
              for d = 1:3
                  E_node_sum(:,d) = E_node_sum(:,d) + ...
                      accumarray(nids, Enode(:,d), [N,1]);
              end
              E_node_cnt = E_node_cnt + accumarray(nids, 1, [N,1]);
          end
      end
    
      E_node    = E_node_sum ./ max(E_node_cnt, 1);
      Emag_node = sqrt(sum(E_node.^2, 2));
      Emag_elem = sqrt(sum(E_elem.^2, 2));
    
      out.E_node    = E_node;
      out.E_elem    = E_elem;
      out.Emag_node = Emag_node;
      out.Emag_elem = Emag_elem;
      end
    
      %  Tet10 shape function derivatives w.r.t. (r,s,t)
      function [dNr, dNs, dNt] = tet10_shape_derivs(r, s, t)
      L1=r; L2=s; L3=t; L4=1-r-s-t;
      dL = [1  0  0  -1;    % dLi/dr
            0  1  0  -1;    % dLi/ds
            0  0  1  -1];   % dLi/dt
      L = [L1; L2; L3; L4];
    
      dNr = zeros(10,1); dNs = zeros(10,1); dNt = zeros(10,1);
    
      % Corner nodes: Ni = Li(2Li - 1),  dNi/dLi = 4Li - 1
      for i = 1:4
          c = 4*L(i) - 1;
          dNr(i) = c*dL(1,i);
          dNs(i) = c*dL(2,i);
          dNt(i) = c*dL(3,i);
      end
    
      % Mid-edge nodes in canonical SF order:
      %   N5=4L1L2, N6=4L2L3, N7=4L3L1, N8=4L1L4, N9=4L2L4, N10=4L3L4
      edges = [1 2; 2 3; 3 1; 1 4; 2 4; 3 4];
      for k = 1:6
          i = edges(k,1); j = edges(k,2);
          dNr(4+k) = 4*(dL(1,i)*L(j) + L(i)*dL(1,j));
          dNs(4+k) = 4*(dL(2,i)*L(j) + L(i)*dL(2,j));
          dNt(4+k) = 4*(dL(3,i)*L(j) + L(i)*dL(3,j));
      end
  end