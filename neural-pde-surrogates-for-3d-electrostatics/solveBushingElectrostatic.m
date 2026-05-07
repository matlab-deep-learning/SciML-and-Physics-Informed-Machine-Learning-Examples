function [R,model] = solveBushingElectrostatic(nFins, finRadiusBase, finRadiusTop, finWidth, tubeRadiusBase, tubeRadiusTop, boreRadius, totalLength, options)
    %solveBushingElectrostatic Electrostatic analysis of a parametric transformer bushing insulator.
    %   [R, model] = solveBushingElectrostatic(...) creates a parametric
    %   bushing insulator geometry, embeds it in an air domain, applies boundary 
    %   conditions, and solves. Replicates the workflow from the
    %   documentation example https://www.mathworks.com/help/pde/ug/electrostatic-analysis-of-transformer-bushing-insulator.html
    %
    %   Inputs:
    %       nFins           - Number of fin-like annular rings
    %       finRadiusBase   - Outer radius of the first (bottom) fin
    %       finRadiusTop    - Outer radius of the last (top) fin
    %       finWidth        - Axial width of each fin
    %       tubeRadiusBase  - Tube outer radius at the bottom
    %       tubeRadiusTop   - Tube outer radius at the top
    %       boreRadius      - Inner bore radius
    %       totalLength     - Total axial length of the bushing insulator
    %
    %   Name-Value Arguments:
    %       DoPlot          - Plot results. Default: false
    %       BushingGeometry - fegeometry object to use instead of creating the
    %                         bushing from parameters. Default: []
    %
    %   Outputs:
    %       R     - ElectrostaticResults object from solve()
    %       model - femodel object with geometry, mesh, and material properties
    %
    %   Boundary conditions:
    %       - Inner bore surface:    Voltage = 10 kV (conductor)
    %       - Flat annular ring:     Voltage = 0 V   (ground)
    %
    %   Material properties:
    %       - Air domain (Cell 1):          Relative permittivity = 1
    %       - Bushing insulator (Cell 2):   Relative permittivity = 5
    
    %   Copyright 2026 The MathWorks, Inc. 

    arguments
        nFins
        finRadiusBase
        finRadiusTop
        finWidth
        tubeRadiusBase
        tubeRadiusTop
        boreRadius
        totalLength
        options.DoPlot = false
        options.BushingGeometry = []
    end

    %% Step 1: Create the parametric bushing geometry (or use provided one)
    if isempty(options.BushingGeometry)
        [gmBushing, ~] = createBushing(nFins, finRadiusBase, finRadiusTop, ...
                                        finWidth, tubeRadiusBase, tubeRadiusTop, ...
                                        boreRadius, totalLength);
    else
        gmBushing = options.BushingGeometry;
    end

    %% Step 2: Create the air domain surrounding the bushing
    % Fixed-size air cuboid matching the documentation example dimensions
    % (1, 0.4, 0.4) but with the long axis along Z to match this example's orientation.
    % Centered in XY, translated so the bushing (Z=0 to totalLength) is
    % fully enclosed.
    gmAir = fegeometry(multicuboid(0.4, 0.4, 1));
    gmAir = translate(gmAir, [0, 0, totalLength/2 - 0.5]);

    %% Step 3: Combine air + bushing
    % Cell 1 = air, Cell 2 = bushing insulator
    gmModel = addCell(gmAir, gmBushing);

    %% Step 4: Find boundary condition faces in the combined geometry
    % Face IDs change after addCell, so re-detect using probe points.

    % Bore face: probe at the bore radius, mid-length
    boreFace = nearestFace(gmModel, [boreRadius, 0, totalLength/2]);

    % Flat annular face: at the peak of the last fin
    % The last fin center is at totalLength - endStub, where endStub = 0.06*totalLength
    lastFinZ = totalLength - 0.06 * totalLength;
    % Probe at mid-radius between tube and fin
    tubeRAtSplit = tubeRadiusBase + (tubeRadiusTop - tubeRadiusBase) * lastFinZ / totalLength;
    midR = (tubeRAtSplit + finRadiusTop) / 2;
    annularFace = nearestFace(gmModel, [midR, 0, lastFinZ]);

    %% Step 5: Set up the electrostatic model
    model = femodel(AnalysisType="electrostatic", Geometry=gmModel);
    model.VacuumPermittivity = 8.8541878128E-12;

    % Material properties
    model.MaterialProperties(1) = materialProperties(RelativePermittivity=1);  % air
    model.MaterialProperties(2) = materialProperties(RelativePermittivity=5);  % insulator

    % Boundary conditions
    model.FaceBC(boreFace) = faceBC(Voltage=10E3);      % 10 kV on conductor
    model.FaceBC(annularFace) = faceBC(Voltage=0);      % ground 

    %% Step 6: Mesh and solve
    model = generateMesh(model, Hmax=0.025);
    R = solve(model);

    %% Step 7 (optional): Visualize results on the bushing
    if options.DoPlot
        elemsBushing = findElements(R.Mesh, "Region", Cell=2);
    
        figure('Position', [100 100 1400 500])
        tiledlayout(1, 2)
    
        % Electric potential
        nexttile
        pdeplot3D(R.Mesh.Nodes, R.Mesh.Elements(:, elemsBushing), ...
                  ColorMapData=R.ElectricPotential);
        title('Electric Potential (V)')
        colorbar
    
        % Electric field magnitude
        nexttile
        Emag = sqrt(R.ElectricField.Ex.^2 + R.ElectricField.Ey.^2 + ...
                    R.ElectricField.Ez.^2);
        pdeplot3D(R.Mesh.Nodes, R.Mesh.Elements(:, elemsBushing), ...
                  ColorMapData=Emag);
        title('Electric Field Magnitude (V/m)')
        colorbar
    end

end
