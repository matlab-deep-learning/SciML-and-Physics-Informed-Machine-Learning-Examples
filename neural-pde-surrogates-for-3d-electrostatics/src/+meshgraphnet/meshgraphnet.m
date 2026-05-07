function x = meshgraphnet(p,x,xe,B,cfg)
    % meshgraphnet Forward pass of a MeshGraphNet.
    %
    %   x = meshgraphnet.meshgraphnet(p, x, xe, B, cfg) runs the
    %   encode-process-decode forward pass of a MeshGraphNet. Node features
    %   are augmented with Fourier positional embeddings, encoded with an
    %   MLP, processed through message-passing layers, and decoded to
    %   produce the output predictions.
    %
    %   Inputs:
    %       p   - Struct of learnable parameters (see meshgraphnet.initialize).
    %       x   - Node feature matrix of size C-by-NumNodes.
    %       xe  - Edge feature matrix of size E-by-NumEdges.
    %       B   - Node-edge adjacency matrix of size NumNodes-by-NumEdges.
    %       cfg - Configuration struct with fields NumLayersEncoder,
    %             NumLayersProcessor, NumLayersDecoder, NumProcessors,
    %             MaxFreq, and OutputSize.
    %
    %   Output:
    %       x - Predicted output of size OutputSize-by-NumNodes.
    
    %   Copyright 2026 The MathWorks, Inc.

    % Add Fourier position embeddings
    x = meshgraphnet.fourierPositionalEmbedding(x,cfg.MaxFreq);
    
    % Encode node and edge features with MLPs
    x = meshgraphnet.mlp(p.node_encoder,x,cfg.NumLayersEncoder);
    xe = meshgraphnet.mlp(p.edge_encoder,xe,cfg.NumLayersEncoder);
    
    % Loop over "processors" - message passing layers between edges and nodes
    z = x;
    for i = 1:cfg.NumProcessors
    
        % Accumulate node features onto edges
        xe = meshgraphnet.nodeToEdgeAccumulation(x,xe,B);
    
        % Apply MLP to edge representations
        xe = meshgraphnet.mlp(p.processors.("edge_processor_"+i),xe,cfg.NumLayersProcessor);
    
        % Accumulate edge features onto nodes
        x = meshgraphnet.edgeToNodeAccumulation(x,xe,B);
    
        % Apply MLP to node representations
        x = meshgraphnet.mlp(p.processors.("node_processor_"+i),x,cfg.NumLayersProcessor);
    end
    
    % Residual connection around the processors
    x = x+z;
    
    % Node decoder
    x = meshgraphnet.mlp(p.node_decoder,x,cfg.NumLayersDecoder);
end