classdef TNNCell <  nnet.layer.Layer
    % TNNCell   Thermal neural network cell
    %
    % TNNCell performs the TNN forward pass for a single time step.

    %   Copyright 2025 The MathWorks, Inc.

    properties
        SampleTime (1,1) double = 0.5; % in seconds
        OutputSize (1,1) double
        IncidenceMatrix_x (:,:) double {mustBeInteger}
        IncidenceMatrix_u (:,:) double {mustBeInteger}
        TemperatureIndices (:,1) double {mustBePositive,mustBeInteger}
        NonTemperatureIndices (:,1) double {mustBePositive,mustBeInteger}
        InputColumns (:,1) string
        TargetColumns (:,1) string
        TemperatureColumns (:,1) string
    end

    properties (Learnable)
        ConductanceNet
        PowerLoss
        Capacitance
    end

    methods

        function this = TNNCell(inputStruct)
            % Construct TNNCell
            this.NumInputs = 2;
            this.OutputSize = length(inputStruct.targetCols);
            nTemps = length(inputStruct.temperatureCols);

            % Build incidence matrices for fully connected graph
            [this.IncidenceMatrix_x, this.IncidenceMatrix_u] = buildIncidenceMatrices(this.OutputSize, this.NumInputs);

            % Store column info
            this.InputColumns = strtrim(string(inputStruct.inputCols))';
            this.TargetColumns = strtrim(string(inputStruct.targetCols))';
            this.TemperatureColumns = strtrim(string(inputStruct.temperatureCols))';

            % Indices for temperature and non-temperature columns
            this.TemperatureIndices = find(ismember(this.InputColumns, this.TemperatureColumns));
            this.NonTemperatureIndices = find(~ismember(this.InputColumns, [this.TemperatureColumns; "profile_id"]));
        end


        function this = generateNetworks(this)
            nTemps = length(this.TemperatureColumns);
            nConds = 0.5 * nTemps * (nTemps - 1) - 1; % fully connected except between the two external nodes
            numNeurons = 16;

            % By default, just use one dense layer + sigmoid activations
            this.ConductanceNet = dlnetwork([featureInputLayer(length(this.InputColumns) + this.OutputSize),...
                fullyConnectedLayer(nConds,Name = "conduc_fc1"),sigmoidLayer]);

            % By default, use two dense layers + tanh activations
            this.PowerLoss = dlnetwork([featureInputLayer(length(this.InputColumns) + this.OutputSize),...
                fullyConnectedLayer(numNeurons,Name = "ploss_fc1"),...
                tanhLayer,...
                fullyConnectedLayer(this.OutputSize,Name="ploss_fc2")]);

            this.Capacitance = dlarray(randn(this.OutputSize, 1,'single') * 0.5 - 9.2); % Initialize caps
        end


        function out = predict(this, input, prevOut)
            % Extract temperatures
            tempsInternal = prevOut; % internal nodes
            tempsExternal = input(this.TemperatureIndices,:); % external nodes
            subNNInput = [input; prevOut];

            E_x = this.IncidenceMatrix_x;
            E_u = this.IncidenceMatrix_u;

            % Conductance network forward pass
            g = abs(predict(this.ConductanceNet, subNNInput'))';

            % Power loss network forward pass
            q = abs(predict(this.PowerLoss, subNNInput'))';

            % Compute temperature differences across edges
            dT = E_x' * tempsInternal + E_u' * tempsExternal;

            % Heat flow on edges
            phi = g .* dT;

            % Net outflow from internal nodes
            netOutflow = E_x * phi;

            % State derivative using incidence-based formulation
            dx = exp(this.Capacitance) .* (-netOutflow + q);

            % Update temperatures
            out = prevOut + this.SampleTime .* dx;

            % Clip output
            out = max(min(out, 5), -1);
        end

    end
end


function [E_x, E_u] = buildIncidenceMatrices(numInternal, numExternal)
% numInternal: number of internal nodes
% numExternal: number of external nodes
% Output:
%   E_x: [numInternal x L] incidence matrix for internal nodes (fully
%   connected graph)
%   E_u: [numExternal x L] incidence matrix for external nodes (fully
%   connected)

% Calculate number of edges for fully connected internal graph
L_internal = nchoosek(numInternal, 2); % fully connected internal nodes
L_external = numInternal * numExternal; % each external node connected to all internal nodes
L = L_internal + L_external;

% Initialize matrices
E_x = zeros(numInternal, L);
E_u = zeros(numExternal, L);

edgeIdx = 1;

% Internal edges (fully connected)
for i = 1:numInternal
    for j = i+1:numInternal
        E_x(i, edgeIdx) = 1;   % source
        E_x(j, edgeIdx) = -1;  % target
        edgeIdx = edgeIdx + 1;
    end
end

% External edges (connect each external node to all internal nodes)
for ext = 1:numExternal
    for int = 1:numInternal
        E_x(int, edgeIdx) = 1;      % internal node as source
        E_u(ext, edgeIdx) = -1;     % external node as target
        edgeIdx = edgeIdx + 1;
    end
end
end
