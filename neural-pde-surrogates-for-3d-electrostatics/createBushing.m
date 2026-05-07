function [gm, faceIDs] = createBushing(nFins, finRadiusBase, finRadiusTop, finWidth, tubeRadiusBase, tubeRadiusTop, boreRadius, totalLength)
    %createBushing Create a parametric transformer bushing insulator geometry.
    %   [gm, faceIDs] = createBushing(nFins, finRadiusBase, finRadiusTop,
    %   finWidth, tubeRadiusBase, tubeRadiusTop, boreRadius, totalLength)
    %   creates an axisymmetric transformer bushing with nFins annular
    %   rings. The fins have a raised-cosine profile and are evenly spaced.
    %   Fin radii are linearly interpolated from finRadiusBase (first fin) to
    %   finRadiusTop (last fin). The tube tapers linearly from tubeRadiusBase
    %   at Z=0 to tubeRadiusTop at Z=totalLength.
    %
    %   The geometry axis is along Z. The last (topmost) fin has a flat annular
    %   face at its peak. The geometry is modeled after the documentation
    %   example: https://www.mathworks.com/help/pde/ug/electrostatic-analysis-of-transformer-bushing-insulator.html
    %
    %   Inputs:
    %       nFins           - Number of annular fin rings
    %       finRadiusBase   - Outer radius of the first (bottom) fin
    %       finRadiusTop    - Outer radius of the last (top) fin
    %       finWidth        - Axial width of each fin (full width of cosine bump)
    %       tubeRadiusBase  - Tube outer radius at Z=0 (bottom)
    %       tubeRadiusTop   - Tube outer radius at Z=totalLength (top)
    %       boreRadius      - Inner bore radius (central hole)
    %       totalLength     - Total axial length of the bushing
    %
    %   Outputs:
    %       gm      - fegeometry object representing transformer bushing insulator geometry
    %       faceIDs - struct with fields:
    %                   .bore        - face ID for the inner bore surface
    %                   .flatAnnular - face ID for the flat annular ring at
    %                                  the top of the last fin

    %   Copyright 2026 The MathWorks, Inc.

    %% Compute fin positions and radii
    % endStub is the distance from the last fin center to the top of the
    % bushing.
    endStub = 0.06 * totalLength;  % short tube stub above the last fin peak
    startMargin = 0.12 * totalLength;  % longer tube section at the bottom
    finPositions = linspace(startMargin + finWidth/2, ...
                            totalLength - endStub, nFins);
    finRadii = linspace(finRadiusBase, finRadiusTop, nFins);

    % Clamp finWidth so adjacent fins don't overlap (leave a small tube gap)
    if nFins > 1
        spacing = finPositions(2) - finPositions(1);
        maxFinWidth = 0.9 * spacing;
        if finWidth > maxFinWidth
            finWidth = maxFinWidth;
        end
    end

    %% Helper: tube radius at any axial position (linear taper)
    tubeRadiusAt = @(z) tubeRadiusBase + (tubeRadiusTop - tubeRadiusBase) * z / totalLength;

    %% Build the outer radius profile
    nArc = 8;  % points per fin bump (raised cosine)

    z_all = [];
    r_all = [];

    % Tube section before first fin
    z_start = linspace(0, finPositions(1) - finWidth/2, 3);
    z_all = z_start;
    r_all = tubeRadiusAt(z_start);

    for i = 1:nFins
        hw = finWidth / 2;
        center = finPositions(i);

        % Raised cosine bump for the fin, sitting on the local tube radius
        zFin = linspace(center - hw, center + hw, nArc);
        localTubeR = tubeRadiusAt(zFin);
        bumpHeight = finRadii(i) - localTubeR;
        rFin = localTubeR + 0.5 .* bumpHeight .* ...
               (1 + cos(pi * (zFin - center) / hw));

        z_all = [z_all, zFin]; 
        r_all = [r_all, rFin]; 

        % Tube section after ith fin
        if i < nFins
            zTube = linspace(center + hw + 0.002, ...
                             finPositions(i+1) - hw - 0.002, 3);
        else
            zTube = linspace(center + hw + 0.002, totalLength, 3);
        end
        z_all = [z_all, zTube]; 
        r_all = [r_all, tubeRadiusAt(zTube)]; 
    end

    % Remove duplicates and sort
    [z_all, ia] = unique(z_all);
    r_all = r_all(ia);

    %% Split at peak of last fin to create the flat annular face
    lastFinCenter = finPositions(end);
    [~, splitIdx] = min(abs(z_all - lastFinCenter));

    %% Revolve lower body (all fins up to peak of last fin)
    xv_low = [0, r_all(1:splitIdx), 0];
    yv_low = [z_all(1), z_all(1:splitIdx), z_all(splitIdx)];
    gm_lower = fegeometry(revolveProfile(xv_low, yv_low));

    %% Revolve upper body (tube stub above last fin)
    % Clamp radii to the local tube radius (remove any residual fin contribution)
    localTubeAtSplit = tubeRadiusAt(z_all(splitIdx));
    r_upper_raw = r_all(splitIdx+1:end);
    r_upper_clamped = min(r_upper_raw, tubeRadiusAt(z_all(splitIdx+1:end)));
    r_upper = [localTubeAtSplit, r_upper_clamped];
    ax_upper = [z_all(splitIdx), z_all(splitIdx+1:end)];
    xv_up = [0, r_upper, 0];
    yv_up = [ax_upper(1), ax_upper, ax_upper(end)];
    gm_upper = fegeometry(revolveProfile(xv_up, yv_up));

    %% Union the two halves and subtract the bore
    gm = union(gm_lower, gm_upper);
    bore = fegeometry(multicylinder(boreRadius, totalLength));
    gm = subtract(gm, bore);

    %% Find face IDs for applying boundary conditions programmatically using nearestFace
    z_split = z_all(splitIdx);
    mid_r = (localTubeAtSplit + r_all(splitIdx)) / 2;
    faceIDs.flatAnnular = nearestFace(gm, [mid_r, 0, z_split]);
    faceIDs.bore = nearestFace(gm, [boreRadius, 0, totalLength/2]);

end

function tri = revolveProfile(rProfile, zProfile, nAngles)
    %revolveProfile Create a triangulated solid of revolution.
    %   tri = revolveProfile(rProfile, zProfile) revolves the closed 2D
    %   polygon defined by (rProfile, zProfile) around the Z axis to produce
    %   a closed triangulated surface mesh. rProfile contains radial
    %   coordinates (>= 0) and zProfile contains axial coordinates.
    %   Vertices with r ≈ 0 are treated as on-axis pole points.
    %
    %   tri = revolveProfile(rProfile, zProfile, nAngles) specifies the
    %   number of angular divisions (default: 72, i.e. 5-degree steps).

    if nargin < 3
        nAngles = 72;
    end

    theta = linspace(0, 2*pi, nAngles + 1);
    theta(end) = []; % remove duplicate at 2*pi

    nPts = numel(rProfile);
    onAxis = rProfile < 1e-10;
    nOff = sum(~onAxis);
    nOn = sum(onAxis);

    % Preallocate vertices
    nVerts = nOff * nAngles + nOn;
    verts = zeros(nVerts, 3);
    vertMap = zeros(nPts, nAngles);

    k = 0;
    for i = 1:nPts
        if onAxis(i)
            k = k + 1;
            verts(k,:) = [0, 0, zProfile(i)];
            vertMap(i,:) = k; % all angles map to same pole vertex
        else
            for j = 1:nAngles
                k = k + 1;
                verts(k,:) = [rProfile(i)*cos(theta(j)), ...
                              rProfile(i)*sin(theta(j)), ...
                              zProfile(i)];
                vertMap(i,j) = k;
            end
        end
    end

    % Preallocate faces (upper bound: 2 triangles per quad)
    faces = zeros(2 * nPts * nAngles, 3);
    f = 0;

    for i = 1:nPts
        inext = mod(i, nPts) + 1;
        for j = 1:nAngles
            jnext = mod(j, nAngles) + 1;

            v1 = vertMap(i, j);
            v2 = vertMap(i, jnext);
            v3 = vertMap(inext, j);
            v4 = vertMap(inext, jnext);

            if v1 == v2 && v3 == v4
                % Both on axis — degenerate, skip
                continue;
            elseif v1 == v2
                % Current point on axis — fan triangle
                f = f + 1;
                faces(f,:) = [v1, v4, v3];
            elseif v3 == v4
                % Next point on axis — fan triangle
                f = f + 1;
                faces(f,:) = [v1, v3, v2];
            else
                % Quad — split into two triangles
                f = f + 1;
                faces(f,:) = [v1, v4, v3];
                f = f + 1;
                faces(f,:) = [v1, v2, v4];
            end
        end
    end

    faces = faces(1:f,:);
    tri = triangulation(faces, verts);
end