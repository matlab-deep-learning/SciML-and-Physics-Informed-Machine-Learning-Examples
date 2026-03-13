classdef depthwiseConv3dLayer < nnet.layer.Layer & ...
        nnet.layer.Formattable & nnet.layer.Acceleratable %#codegen
%DEPTHWISECONV3DLAYER - Depthwise 3D convolution layer for channel-wise scaling.
%  This layer performs element-wise multiplication of input channels with
%  learnable weights, scaling each channel independently.
%  It supports optional bias addition and is designed for use in neural
%  network architectures requiring channel-wise feature modulation.
%
%  Given input X of size (S, S, S, C, B), returns X .* W
%  where W is of size (1, 1, 1, C, 1).

% Copyright 2026 The MathWorks, Inc.
    
    properties (Learnable)
        Weight
        Bias
    end
    
    properties
        NumChannels
        UseBias
    end  
    
    methods
        function layer = depthwiseConv3dLayer(numChannels, args)
            arguments
                numChannels (1, 1) double
                args.UseBias (1, 1) logical = false
                args.Name (1, 1) string = "depthwiseConv"
            end
            
            layer.NumChannels = numChannels;
            layer.UseBias = args.UseBias;
            layer.Name = args.Name;
        end

        function layer = initialize(layer, layout)
            if ~isempty(layer.Weight)
                return
            end
            sdim = finddim(layout, 'S');
            cdim = finddim(layout, 'C');
            tdim = finddim(layout, 'T');

            hasTimeDimension = ~isempty(tdim);
            numSpatialDimensions  = numel(sdim); 
            tfno.validation.assertValidNumConvolutionDimensions(3, hasTimeDimension, numSpatialDimensions);

            % Check the input data has a channel dimension
            tfno.validation.assertInputHasChannelDim(3, cdim);

            % There are either SSSCB or SSTCB dims in unknown order
            weightSize = ones(1, 5);
            weightSize(cdim) = layout.Size(cdim);

            % Glorot initialization
            Z = 2*rand(weightSize, 'single') - 1;
            bound = sqrt(6 / 2);
            layer.Weight = dlarray(bound * Z);
            
            if layer.UseBias
                layer.Bias = dlarray(zeros(weightSize, 'single'), layout.Format);
            else
                layer.Bias = [];
            end
        end

        function Z = predict(layer, X)
            Z = X .* layer.Weight;
            if layer.UseBias
                Z = Z + layer.Bias;
            end
        end
    end
end