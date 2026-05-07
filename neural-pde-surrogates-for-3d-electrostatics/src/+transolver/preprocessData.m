function [inputs,targets,efieldTargets] = preprocessData(data,stride)
    % preprocessData Extract and stride input, potential, and E-field data.
    %
    %   [inputs, targets, efieldTargets] = transolver.preprocessData(data,
    %   stride) extracts coordinate and material inputs, electric potential
    %   targets, and electric field targets from a struct array of
    %   simulation data. Nodes with NaN values are removed and the
    %   remaining nodes are subsampled by stride.
    %
    %   Inputs:
    %       data   - Struct array with fields Coord, Epsilon, Potential,
    %                and ElectricField.
    %       stride - Subsampling stride for node selection.
    %
    %   Outputs:
    %       inputs       - Cell array of input matrices (4-by-Ni each).
    %       targets      - Cell array of potential target vectors (1-by-Ni).
    %       efieldTargets - Cell array of E-field matrices (3-by-Ni).
    
    %   Copyright 2026 The MathWorks, Inc.
    
    inputs = cell(numel(data),1);
    targets = cell(numel(data),1);
    efieldTargets = cell(numel(data),1);

        function [input,target,efield] = extractInputAndTarget(data,stride)

            input = cat(1, ...
                data.Coord,...
                data.Epsilon');

            % electric potential target
            target = data.Potential';

            % electric field target (3 x N)
            efield = data.ElectricField;

            % Find any nodes with a NaN input, target, or E-field and remove
            nanInput = isnan(input);
            nanTarget = isnan(target);
            nanEfield = isnan(efield);
            nanNode = any(nanInput,1) | any(nanTarget,1) | any(nanEfield,1);
            input(:,nanNode) = [];
            target(:,nanNode) = [];
            efield(:,nanNode) = [];
            input = input(:,1:stride:end);
            target = target(:,1:stride:end);
            efield = efield(:,1:stride:end);
        end

    % Loop over data
    for i = 1:numel(data)
        [inputs{i},targets{i},efieldTargets{i}] = extractInputAndTarget(data(i),stride);
    end
end