%[text] # Generate Training Data (Geometry + 3D Simulation Dataset)
%[text] This script generates a paired dataset of **3D geometries** and **corresponding electrostatic simulation results** on those geometries for training and evaluating a deep learning surrogate model. 
%[text] Each sample consists of:
%[text] - A parameterized transformer bushing insulator-like geometry (***Note***: the geometry parameters are only used to procedurally generate CAD geometry files and ensure reproducibility of the dataset. They are not provided as inputs to the surrogate model during training or inference. This mimics a common scenario where you may have access to the CAD geometry files, but no underlying parametric description of how those shapes were created.)
%[text] - A finite element solution of the electrostatic potential field and the derived electric field
%[text] - Per-node material properties
%[text] - Mesh information  \
%[text] The resulting dataset is intended for full-field surrogate modeling, where the model learns to predict spatially varying physical fields given geometry and material inputs.
%[text] The data generation takes several minutes to complete (~5 minutes using Parallel pool with 8 workers). 
N = 75; % number of geometries to generate

% Folders where generated data will be stored
projectRoot = findProjectRoot("startup.m");
stldir = fullfile(projectRoot,"STL");
datadir = fullfile(projectRoot,"data");
%%
%[text] ## Generate CAD Data
%[text] This step creates a collection of parameterized axisymmetric 3D geometries representing variations of transformer bushing insulators. Each geometry consists of a central hollow tapered tube and a stack of annular fins, linearly increasing in radius from top to bottom. The geometry parameters are:
%[text] - number of fins
%[text] - fin radius at the base (smallest fin is at the bottom)
%[text] - fin radius at the top (largest fin is at the top)
%[text] - fin width
%[text] - tube radius at the base (narrower at the bottom)
%[text] - tube radius at the top (wider at the top)
%[text] - bore radius (fixed)
%[text] - total length (fixed) \
%[text] The varying geometric parameters are sampled within prespecified bounds using Latin Hypercube Sampling (LHS) to efficiently cover the parameter space. 
%[text] Define the parameter ranges.
% Varying parameters
nFinsRange       = [6, 12];       % integer
finRadBaseRange  = [0.06, 0.10];  % meters
finRadTopRange   = [0.10, 0.16];  % meters
finWidthRange    = [0.012, 0.05]; % meters
tubeRadBaseRange = [0.025, 0.045]; % meters (tube bottom)
tubeRadTopRange  = [0.045, 0.065]; % meters (tube top)

% Fixed parameters
boreRadius  = 0.015;
totalLength = 0.6;
%[text] Generate the bushing parameters using LHS. Uses [lhsdesign](https://www.mathworks.com/help/releases/R2026a/stats/lhsdesign.html?searchPort=58037) from Statistics and Machine Learning Toolbox™.  
rng(42)  % reproducibility
lhs = lhsdesign(N, 6);

nFinsAll       = round(nFinsRange(1) + lhs(:,1) * diff(nFinsRange));
finRadBaseAll  = finRadBaseRange(1)  + lhs(:,2) * diff(finRadBaseRange);
finRadTopAll   = finRadTopRange(1)   + lhs(:,3) * diff(finRadTopRange);
finWidthAll    = finWidthRange(1)    + lhs(:,4) * diff(finWidthRange);
tubeRadBaseAll = tubeRadBaseRange(1) + lhs(:,5) * diff(tubeRadBaseRange);
tubeRadTopAll  = tubeRadTopRange(1)  + lhs(:,6) * diff(tubeRadTopRange);

% Ensure finRadiusTop >= finRadiusBase for each sample
finRadTopAll = max(finRadTopAll, finRadBaseAll + 0.01);

% Ensure tube top >= tube base
tubeRadTopAll = max(tubeRadTopAll, tubeRadBaseAll);
%%
%[text] Construct and export the geometries. Construct each geometry using primitive and Boolean operations. See `createBushing.m` for the geometry construction. 
%[text] After the geometry is created, it is meshed and its surface triangulation is extracted. These triangulations are saved as STL files. 
geometries = cell(N, 1);
faceIDsAll = cell(N, 1);
if isempty(gcp("nocreate"))
    parpool; % use default cluster profile
end
parfor i = 1:N %[output:group:7298d8dd] %[output:4e79d9bc]
    fprintf('Variant %d: nFins=%d, finRad=[%.3f,%.3f], width=%.3f, tubeRad=[%.3f,%.3f]\n', ...
        i, nFinsAll(i), finRadBaseAll(i), finRadTopAll(i), finWidthAll(i), ...
        tubeRadBaseAll(i), tubeRadTopAll(i));

    [gm, fIDs] = createBushing(nFinsAll(i), finRadBaseAll(i), finRadTopAll(i), ...
        finWidthAll(i), tubeRadBaseAll(i), tubeRadTopAll(i), ...
        boreRadius, totalLength);

    geometries{i} = gm;
    faceIDsAll{i} = fIDs;

    % Export to STL and save
    filename = sprintf('%s/transformer_bushing_%03d.stl', stldir, i);
    gm = generateMesh(gm,GeometricOrder="linear",Hmax=0.01);

    nodes = gm.Mesh.Nodes;
    elements = gm.Mesh.Elements;

    [f, v] = freeBoundary(triangulation(elements', nodes'));
    TR = triangulation(f, v);
    stlwrite(TR,filename);
end %[output:group:7298d8dd]
%%
%[text] Save the parameters to a table for reproducibility. 
params = table(nFinsAll, finRadBaseAll, finRadTopAll, finWidthAll, ...
    tubeRadBaseAll, tubeRadTopAll, ...
    repmat(boreRadius, N, 1), repmat(totalLength, N, 1), ...
    VariableNames=["NFins", "FinRadiusBase", "FinRadiusTop", "FinWidth", ...
    "TubeRadiusBase", "TubeRadiusTop", "BoreRadius", "TotalLength"]);
save("bushingParams.mat", "params")
fprintf('\nSaved %d parameter sets to bushingParams.mat\n', N); %[output:2096060e]
%%
%[text] Create one extra "inference-only" geometry with no associated simulation data, to demonstrate how the AI model can be used to make inference on new designs for which no high-fidelity simulation data exists. 
rng(2); % for reproducibility
nFinsTest = randi(nFinsRange);
finRadBaseTest = (finRadBaseRange(2)-finRadBaseRange(1))*rand(1) + finRadBaseRange(1);
finRadTopTest = (finRadTopRange(2)-finRadTopRange(1))*rand(1) + finRadTopRange(1);
finWidthTest = (finWidthRange(2)-finWidthRange(1))*rand(1) + finWidthRange(1);
tubeRadBaseTest = (tubeRadBaseRange(2)-tubeRadBaseRange(1))*rand(1) + tubeRadBaseRange(1);
tubeRadTopTest = (tubeRadTopRange(2)-tubeRadTopRange(1))*rand(1) + tubeRadTopRange(1);

gmTest = createBushing(nFinsTest, finRadBaseTest, finRadTopTest, ...
    finWidthTest, tubeRadBaseTest, tubeRadTopTest, ...
    boreRadius, totalLength);

% Save parameters to a table
paramsTest = table(nFinsTest, finRadBaseTest, finRadTopTest, ...
    finWidthTest, tubeRadBaseTest, tubeRadTopTest, ...
    boreRadius, totalLength, ...
    VariableNames=["NFins", "FinRadiusBase", "FinRadiusTop", "FinWidth", ...
    "TubeRadiusBase", "TubeRadiusTop", "BoreRadius", "TotalLength"]);
save("testParams.mat", "paramsTest")
fprintf('\nSaved test parameter set to testParams.mat\n'); %[output:9651a736]

% Export to STL and save
filename = sprintf('%s/TEST_bushing.stl', stldir);
gmTest = generateMesh(gmTest,GeometricOrder="linear",Hmax=0.01);

nodes = gmTest.Mesh.Nodes;
elements = gmTest.Mesh.Elements;

[f, v] = freeBoundary(triangulation(elements', nodes'));
TR = triangulation(f, v);
stlwrite(TR,filename);
%%
%[text] ## Generate CAE data (electrostatic simulation)
%[text] For each geometry, a 3D electrostatic finite element simulation is performed with PDE Toolbox. The simulation setup follows the documentation example [Electrostatic Analysis of Transformer Bushing Insulator](https://www.mathworks.com/help/pde/ug/electrostatic-analysis-of-transformer-bushing-insulator.html). 
%[text] In the simulation setup, we embed the insulator geometry in a surrounding air volume. The materials used are:
%[text] - Air (relative permittivity = 1)
%[text] - Insulator (relative permittivity = 5) \
%[text] The boundary conditions used are: 
%[text] - Fixed high voltage applied to the inner bore surface,
%[text] - Ground applied to the topmost, flat annular ring.  \
%[text] To automate application of the boundary conditions, we identify the faces using [`nearestFace`](https://www.mathworks.com/help/pde/ug/discretegeometry.nearestface.html). For the full electrostatic simulation, see `solveBushingElectrostatic.m`.
%[text] For each geometry, the following simulation data is saved to a `.mat` file:
%[text] - Nodal coordinates (3xN array)
%[text] - Per-node relative permittivity values (1xN array)
%[text] - Per-node electric potential values (1xN array)
%[text] - Per-node electric field component values, derived from the FEA solution (3xN array)
%[text] - FEA mesh ([`FEMesh`](https://www.mathworks.com/help/pde/ug/pde.femesh.html) object) \
%[text] This data is designed to support the creation of Transolver and MeshGraphNet models. 
parfor i=1:N
    gmBushing = fegeometry(sprintf('%s/transformer_bushing_%03d.stl', stldir, i));
    p = params(i,:);
    [R,model] = solveBushingElectrostatic(p.NFins, p.FinRadiusBase, p.FinRadiusTop, ...
        p.FinWidth, p.TubeRadiusBase, p.TubeRadiusTop, p.BoreRadius, p.TotalLength, ...
        BushingGeometry=gmBushing);
    % Save results in temporary struct
    Di = struct();
    % Get the mesh nodes
    Di.Coord = R.Mesh.Nodes; % 3 x numNodes array
    % Get the material properties vector
    epsilon = zeros(size(R.Mesh.Nodes,2),1);
    idxAir = findNodes(R.Mesh,"Region",Cell=1);
    epsilon(idxAir) = model.MaterialProperties(1).RelativePermittivity;
    idxBushing = findNodes(R.Mesh,"Region",Cell=2);
    epsilon(idxBushing) = model.MaterialProperties(2).RelativePermittivity;
    Di.Epsilon = epsilon; 
    % Get the targets
    Di.Potential = R.ElectricPotential; % numNodes x 1
    Di.ElectricField = [R.ElectricField.Ex'; R.ElectricField.Ey'; R.ElectricField.Ez']; % 3 x numNodes
    % Get the mesh data
    Di.Mesh = R.Mesh;    
    % Save the results to a .mat file for further analysis
    Ssave = struct("D",Di);
    save(sprintf('%s/EM_results_%03d.mat',datadir,i), "-fromstruct", Ssave);
end

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"onright"}
%---
%[output:4e79d9bc]
%   data: {"dataType":"text","outputData":{"text":"Variant 4: nFins=10, finRad=[0.078,0.143], width=0.024, tubeRad=[0.032,0.049]\nVariant 3: nFins=7, finRad=[0.063,0.152], width=0.025, tubeRad=[0.025,0.064]\nVariant 8: nFins=8, finRad=[0.066,0.123], width=0.015, tubeRad=[0.034,0.057]\nVariant 1: nFins=7, finRad=[0.071,0.147], width=0.029, tubeRad=[0.040,0.049]\nVariant 2: nFins=6, finRad=[0.081,0.129], width=0.016, tubeRad=[0.026,0.051]\nVariant 5: nFins=9, finRad=[0.082,0.160], width=0.020, tubeRad=[0.036,0.055]\nVariant 6: nFins=11, finRad=[0.074,0.120], width=0.038, tubeRad=[0.040,0.057]\nVariant 7: nFins=11, finRad=[0.089,0.140], width=0.026, tubeRad=[0.039,0.059]\nVariant 12: nFins=8, finRad=[0.084,0.156], width=0.044, tubeRad=[0.034,0.048]\nVariant 40: nFins=10, finRad=[0.087,0.114], width=0.019, tubeRad=[0.029,0.054]\nVariant 16: nFins=9, finRad=[0.090,0.138], width=0.042, tubeRad=[0.030,0.054]\nVariant 24: nFins=11, finRad=[0.098,0.108], width=0.044, tubeRad=[0.044,0.057]\nVariant 28: nFins=9, finRad=[0.094,0.137], width=0.036, tubeRad=[0.031,0.047]\nVariant 20: nFins=7, finRad=[0.075,0.117], width=0.021, tubeRad=[0.030,0.061]\nVariant 32: nFins=11, finRad=[0.085,0.103], width=0.019, tubeRad=[0.030,0.060]\nVariant 36: nFins=8, finRad=[0.068,0.127], width=0.042, tubeRad=[0.032,0.048]\nVariant 11: nFins=10, finRad=[0.078,0.146], width=0.015, tubeRad=[0.033,0.055]\nVariant 19: nFins=11, finRad=[0.081,0.121], width=0.029, tubeRad=[0.038,0.059]\nVariant 15: nFins=9, finRad=[0.091,0.125], width=0.030, tubeRad=[0.039,0.047]\nVariant 39: nFins=7, finRad=[0.062,0.158], width=0.038, tubeRad=[0.044,0.062]\nVariant 27: nFins=8, finRad=[0.079,0.121], width=0.043, tubeRad=[0.033,0.064]\nVariant 35: nFins=7, finRad=[0.077,0.153], width=0.013, tubeRad=[0.038,0.062]\nVariant 31: nFins=7, finRad=[0.084,0.142], width=0.039, tubeRad=[0.043,0.063]\nVariant 23: nFins=11, finRad=[0.066,0.101], width=0.034, tubeRad=[0.034,0.048]\nVariant 10: nFins=8, finRad=[0.088,0.134], width=0.035, tubeRad=[0.045,0.052]\nVariant 38: nFins=12, finRad=[0.098,0.108], width=0.022, tubeRad=[0.035,0.056]\nVariant 14: nFins=9, finRad=[0.070,0.125], width=0.025, tubeRad=[0.041,0.046]\nVariant 18: nFins=12, finRad=[0.071,0.104], width=0.034, tubeRad=[0.038,0.057]\nVariant 34: nFins=9, finRad=[0.071,0.149], width=0.046, tubeRad=[0.026,0.053]\nVariant 26: nFins=6, finRad=[0.068,0.126], width=0.037, tubeRad=[0.038,0.046]\nVariant 30: nFins=8, finRad=[0.091,0.154], width=0.031, tubeRad=[0.033,0.059]\nVariant 22: nFins=9, finRad=[0.096,0.114], width=0.041, tubeRad=[0.028,0.050]\nVariant 9: nFins=8, finRad=[0.064,0.128], width=0.019, tubeRad=[0.031,0.045]\nVariant 13: nFins=10, finRad=[0.076,0.140], width=0.026, tubeRad=[0.028,0.053]\nVariant 25: nFins=9, finRad=[0.076,0.157], width=0.044, tubeRad=[0.033,0.059]\nVariant 37: nFins=6, finRad=[0.100,0.133], width=0.017, tubeRad=[0.043,0.063]\nVariant 29: nFins=10, finRad=[0.088,0.110], width=0.035, tubeRad=[0.045,0.056]\nVariant 17: nFins=8, finRad=[0.086,0.115], width=0.024, tubeRad=[0.027,0.049]\nVariant 33: nFins=12, finRad=[0.093,0.123], width=0.047, tubeRad=[0.029,0.062]\nVariant 21: nFins=11, finRad=[0.065,0.118], width=0.046, tubeRad=[0.044,0.050]\nVariant 42: nFins=10, finRad=[0.067,0.135], width=0.042, tubeRad=[0.041,0.055]\nVariant 44: nFins=11, finRad=[0.095,0.129], width=0.012, tubeRad=[0.042,0.048]\nVariant 46: nFins=6, finRad=[0.086,0.108], width=0.031, tubeRad=[0.038,0.052]\nVariant 50: nFins=10, finRad=[0.098,0.119], width=0.033, tubeRad=[0.037,0.064]\nVariant 52: nFins=11, finRad=[0.093,0.156], width=0.040, tubeRad=[0.036,0.050]\nVariant 54: nFins=12, finRad=[0.073,0.150], width=0.040, tubeRad=[0.028,0.058]\nVariant 48: nFins=12, finRad=[0.096,0.109], width=0.032, tubeRad=[0.026,0.063]\nVariant 56: nFins=9, finRad=[0.062,0.108], width=0.048, tubeRad=[0.034,0.056]\nVariant 41: nFins=8, finRad=[0.097,0.152], width=0.041, tubeRad=[0.043,0.058]\nVariant 45: nFins=7, finRad=[0.061,0.113], width=0.021, tubeRad=[0.040,0.047]\nVariant 43: nFins=10, finRad=[0.080,0.132], width=0.027, tubeRad=[0.041,0.061]\nVariant 49: nFins=11, finRad=[0.099,0.142], width=0.020, tubeRad=[0.043,0.051]\nVariant 51: nFins=10, finRad=[0.077,0.102], width=0.027, tubeRad=[0.042,0.052]\nVariant 53: nFins=7, finRad=[0.092,0.149], width=0.017, tubeRad=[0.028,0.061]\nVariant 47: nFins=7, finRad=[0.072,0.111], width=0.050, tubeRad=[0.028,0.049]\nVariant 55: nFins=9, finRad=[0.092,0.159], width=0.028, tubeRad=[0.039,0.055]\nVariant 59: nFins=7, finRad=[0.066,0.137], width=0.047, tubeRad=[0.027,0.065]\nVariant 57: nFins=9, finRad=[0.082,0.122], width=0.023, tubeRad=[0.029,0.060]\nVariant 58: nFins=6, finRad=[0.069,0.139], width=0.032, tubeRad=[0.036,0.051]\nVariant 60: nFins=11, finRad=[0.065,0.106], width=0.016, tubeRad=[0.037,0.064]\nVariant 62: nFins=10, finRad=[0.075,0.113], width=0.036, tubeRad=[0.042,0.050]\nVariant 63: nFins=7, finRad=[0.087,0.111], width=0.014, tubeRad=[0.031,0.058]\nVariant 61: nFins=7, finRad=[0.086,0.145], width=0.048, tubeRad=[0.030,0.058]\nVariant 64: nFins=9, finRad=[0.064,0.130], width=0.022, tubeRad=[0.035,0.047]\nVariant 65: nFins=8, finRad=[0.073,0.131], width=0.036, tubeRad=[0.025,0.045]\nVariant 66: nFins=8, finRad=[0.079,0.131], width=0.014, tubeRad=[0.027,0.052]\nVariant 67: nFins=11, finRad=[0.090,0.103], width=0.029, tubeRad=[0.032,0.054]\nVariant 71: nFins=10, finRad=[0.083,0.150], width=0.045, tubeRad=[0.040,0.054]\nVariant 68: nFins=8, finRad=[0.073,0.148], width=0.023, tubeRad=[0.035,0.062]\nVariant 69: nFins=10, finRad=[0.083,0.116], width=0.031, tubeRad=[0.037,0.060]\nVariant 70: nFins=9, finRad=[0.095,0.105], width=0.013, tubeRad=[0.031,0.061]\nVariant 72: nFins=12, finRad=[0.069,0.155], width=0.049, tubeRad=[0.043,0.053]\nVariant 73: nFins=11, finRad=[0.061,0.145], width=0.049, tubeRad=[0.041,0.054]\nVariant 74: nFins=7, finRad=[0.094,0.143], width=0.038, tubeRad=[0.027,0.065]\nVariant 75: nFins=6, finRad=[0.060,0.136], width=0.018, tubeRad=[0.035,0.046]\n","truncated":false}}
%---
%[output:2096060e]
%   data: {"dataType":"text","outputData":{"text":"\nSaved 75 parameter sets to bushingParams.mat\n","truncated":false}}
%---
%[output:9651a736]
%   data: {"dataType":"text","outputData":{"text":"\nSaved test parameter set to testParams.mat\n","truncated":false}}
%---
