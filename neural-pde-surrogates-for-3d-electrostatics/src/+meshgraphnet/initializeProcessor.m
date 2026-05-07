function p = initializeProcessor(cfg)
    % initializeProcessor Initialize parameters for message-passing processor layers.
    %
    %   p = meshgraphnet.initializeProcessor(cfg) creates a struct
    %   containing node and edge processor MLP parameters for each
    %   message-passing step. The number of processors and their layer
    %   configuration are determined by cfg.
    %
    %   Input:
    %       cfg - Configuration struct with fields NumProcessors,
    %             HiddenSize, and NumLayersProcessor.
    %
    %   Output:
    %       p - Struct with fields node_processor_1, edge_processor_1, ...,
    %           each containing MLP parameters (see initializeMLP).

    %   Copyright 2026 The MathWorks, Inc.
    p = struct();
    for i = 1:cfg.NumProcessors
        p.("node_processor_"+i) = meshgraphnet.initializeMLP(cfg.HiddenSize,cfg.HiddenSize,cfg.NumLayersProcessor);
        p.("edge_processor_"+i) = meshgraphnet.initializeMLP(cfg.HiddenSize,cfg.HiddenSize,cfg.NumLayersProcessor);
    end
end