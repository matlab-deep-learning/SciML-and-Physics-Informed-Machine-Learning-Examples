classdef DiffEqLayer <  nnet.layer.Layer & nnet.layer.Formattable
    % DiffEqLayer   Differential equation layer
    % 
    % This layer holds a TNNCell instance, and performs prediction over
    % tbptt_size time steps.
    % 
    %   Copyright 2025 The MathWorks, Inc.

    properties (Learnable)
        Cell
    end

    methods
        function obj = DiffEqLayer(cell)
            % Constructor to store the cell
            obj.Cell = cell;
            obj.NumInputs = 2;
            obj.NumOutputs = 2;
        end

        function [outputs, state] = predict(this, input, state)
            % Initialize cell array for outputs
            numSteps = size(input, 3); % Assuming input is [features, batch, time]

            % Preallocate the outputs:
            outputs = dlarray(zeros(size(state,[1 2 3]),'single'),'CBT');
            
            % Iterate over each time step
            thisCell = this.Cell;
            for tt = 1:numSteps
                outputs(:,:,tt) = predict(thisCell,squeeze(input(:, :, tt)),state);
                state = squeeze(outputs(:, :, tt));
            end
        end
    end
end


