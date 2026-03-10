classdef optimizedTNNLayer <  nnet.layer.Layer ...
        & nnet.layer.Formattable ...
        & nnet.layer.Acceleratable
    % optimizedTNNLayer   Thermal neural network layer

    %   Copyright 2025 The MathWorks, Inc.

    properties
        SampleTime (1,1) double = 0.5; % in seconds
        OutputSize (1,1) double
        AdjacencyMatrix (:,:) double {mustBePositive,mustBeInteger}
        TemperatureIndices (:,1) double {mustBePositive,mustBeInteger}
        NonTemperatureIndices (:,1) double {mustBePositive,mustBeInteger}
        InputColumns (:,1) string
        TargetColumns (:,1) string
        TemperatureColumns (:,1) string
    end

    properties (Learnable)
        WeightsConductance
        BiasConductance
        WeightsPowerLoss1
        BiasPowerLoss1
        WeightsPowerLoss2
        BiasPowerLoss2
        Capacitance
    end

    methods
        function this = optimizedTNNLayer(inputStruct)
            this.NumInputs = 2;
            this.NumOutputs = 2;
            this.OutputSize = length(inputStruct.targetCols);
            nTemps = length(inputStruct.temperatureCols);

            % Populate adjacency matrix
            this.AdjacencyMatrix = ones(nTemps, nTemps);
            trilIdx = find(tril(ones(nTemps),-1));
            adjIdxArr = 1:(0.5*nTemps*(nTemps-1));
            this.AdjacencyMatrix(trilIdx) = adjIdxArr;
            this.AdjacencyMatrix = this.AdjacencyMatrix + this.AdjacencyMatrix'-1;
            this.AdjacencyMatrix = this.AdjacencyMatrix(1:this.OutputSize, :);

            this.InputColumns = strtrim(string(inputStruct.inputCols))';
            this.TargetColumns = strtrim(string(inputStruct.targetCols))';
            this.TemperatureColumns = strtrim(string(inputStruct.temperatureCols))';

            % Indices for temperature and non-temperature columns
            this.TemperatureIndices = find(ismember(this.InputColumns, this.TemperatureColumns));
            this.NonTemperatureIndices = find(~ismember(this.InputColumns, [this.TemperatureColumns; "profile_id"]));
            this = initAllLearnables(this);
        end

        function this = initAllLearnables(this)

            nTemps = length(this.TemperatureColumns);
            nConds = 0.5 * nTemps * (nTemps - 1);
            numNeurons = 16;
            
            % per default, just use one dense layer + sigmoid activations
            this.WeightsConductance = initializeGlorot(nConds,length(this.InputColumns) + this.OutputSize);
            this.BiasConductance = dlarray(zeros(nConds,1,'single'));

            % per default, use two dense layers + tanh activations
            this.WeightsPowerLoss1 = initializeGlorot(numNeurons,length(this.InputColumns) + this.OutputSize);
            this.BiasPowerLoss1 = dlarray(zeros(numNeurons,1,'single'));
            this.WeightsPowerLoss2 = initializeGlorot(this.OutputSize,numNeurons);
            this.BiasPowerLoss2 = dlarray(zeros(this.OutputSize,1,'single'));

            this.Capacitance = dlarray(randn(this.OutputSize, 1,'single') * 0.5 - 9.2); % Initialize caps
        end

        function [X, h] = predict(this, X, h)
            fcn = tnnWithBackward(this.AdjacencyMatrix, this.SampleTime, this.TemperatureIndices);
            [X, h] = fcn(X, h, this.WeightsConductance, this.BiasConductance, this.WeightsPowerLoss1, this.BiasPowerLoss1, this.WeightsPowerLoss2, this.BiasPowerLoss2, this.Capacitance);
            X = dlarray(X,'CBT');
            h = dlarray(h, 'CBT');
        end

    end
end

function W = initializeGlorot(numNeurons,numInputs)
W = dlarray(sqrt(6./(numInputs+numNeurons))*(2*rand(numNeurons,numInputs,'single')-1));
end
