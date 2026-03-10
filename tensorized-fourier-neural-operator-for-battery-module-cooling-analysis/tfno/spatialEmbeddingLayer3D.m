classdef spatialEmbeddingLayer3D < nnet.layer.Layer & ...
        nnet.layer.Formattable & nnet.layer.Acceleratable %#codegen
%SPATIALEMBEDDINGLAYER3D - 3D spatial embedding layer.
%  layer = SPATIALEMBEDDINGLAYER3D(spatialLimits) creates a 3D spatial
%  embedding layer that adds position embeddings to input data based on
%  specified spatial limits. The layer generates position values linearly
%  spaced between the limits and is discretization invariant. It is returned
%  as a layer object.
%
%  Inputs:
%    spatialLimits - 3x2 matrix specifying [min, max] spatial bounds for each dimension
%
%  Supported Name-Value pairs are:
%    "Name"                    Name for the layer (default: "depthwiseConv")

% Copyright 2026 The MathWorks, Inc.
    
    properties
        SpatialLimits
    end  
    
    methods
        function layer = spatialEmbeddingLayer3D(spatialLimits, args)
            arguments
                spatialLimits (3, 2) double
                args.Name (1, 1) string = "depthwiseConv"
            end
            
            layer.SpatialLimits = spatialLimits;
            layer.Name = args.Name;
        end

        function Z = predict(layer, X)
            sdim = finddim(X, 'S');
            sSize = size(X, sdim);

            S1 = linspace(layer.SpatialLimits(1, 1), ...
                layer.SpatialLimits(1, 2), sSize(1));
            S2 = linspace(layer.SpatialLimits(2, 1), ...
                layer.SpatialLimits(2, 2), sSize(2));
            S3 = linspace(layer.SpatialLimits(3, 1), ...
                layer.SpatialLimits(3, 2), sSize(3));

            [embedding1, embedding2, embedding3] = meshgrid(S1, S2, S3);

            embedding = zeros([sSize, 3], Like=X);
            embedding(:, :, :, 1) = embedding1;
            embedding(:, :, :, 2) = embedding2;
            embedding(:, :, :, 3) = embedding3;

            bdim = finddim(X, 'B');
            bSize = size(X, bdim);
            Z = repmat(embedding, [1, 1, 1, 1, bSize]);
            Z = dlarray(Z, "SSSCB");
        end
    end
end
