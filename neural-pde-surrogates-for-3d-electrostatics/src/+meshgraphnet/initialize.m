function p = initialize(cfg)
    % initialize Initialize all learnable parameters for a MeshGraphNet.
    %
    %   p = meshgraphnet.initialize(cfg) creates and returns a struct of
    %   dlarray parameters for the full MeshGraphNet, including node
    %   encoder, edge encoder, message-passing processors, and node decoder.
    %
    %   Input:
    %       cfg - Configuration struct with fields NodeInputSize,
    %             EdgeInputSize, HiddenSize, OutputSize, MaxFreq,
    %             NumLayersEncoder, NumLayersProcessor, NumLayersDecoder,
    %             and NumProcessors.
    %
    %   Output:
    %       p - Struct of dlarray parameters with fields node_encoder,
    %           edge_encoder, processors, and node_decoder.

    %   Copyright 2026 The MathWorks, Inc.
    p.node_encoder = meshgraphnet.initializeMLP(cfg.NodeInputSize*(2*cfg.MaxFreq+1),cfg.HiddenSize,cfg.NumLayersEncoder);
    p.edge_encoder = meshgraphnet.initializeMLP(cfg.EdgeInputSize,cfg.HiddenSize,cfg.NumLayersEncoder);
    p.processors = meshgraphnet.initializeProcessor(cfg);
    p.node_decoder = meshgraphnet.initializeMLP(cfg.HiddenSize,cfg.HiddenSize,cfg.NumLayersDecoder,cfg.OutputSize);
    p = dlupdate(@dlarray,p);
end