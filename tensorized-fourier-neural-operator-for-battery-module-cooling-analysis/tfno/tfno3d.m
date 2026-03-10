function net = tfno3d(numModes, latentChannelSize, args)
%TFNO3D - Create a 3D Fourier Neural Operator (FNO) network.
%  net = TFNO3D(numModes, latentChannelSize) creates a 3D FNO network with
%  the specified number of Fourier modes and latent channel size.
%
%  Supported Name-Value pairs are:
%    "NumBlocks"              Number of FNO blocks (default: 1)
%    "ExpandNet"              Whether to expand network layers (default: true)
%    "LiftingChannelRatio"    Ratio for lifting channels (default: 2)
%    "ProjectionChannelRatio" Ratio for projection channels (default: 2)
%    "InChannels"             Number of input channels (default: 1)
%    "OutChannels"            Number of output channels (default: 1)
%    "SpatialLimits"          Spatial domain limits [min1, max1; min2, max2; min3, max3] (default: [0, 1; 0, 1; 0, 1])
%    "SpectralRank"           Rank for spectral convolution (0-1) (default: 1)
%    "LinearFNOSkip"          Enable linear skip connection (default: true)
%    "ChannelMLPSkip"         Enable channel MLP skip connection (default: true)
%    "UseSpectralConvBias"    Use bias in spectral convolution (default: true)
%    "FuseSpectralConv"       Directly contract spectral activations with tensorized weights (default: true)
%
% The network architecture consists of:
%   1. Input layer with spatial embedding
%   2. Lifting layers to project input to latent space
%   3. FNO blocks for spectral processing
%   4. Projection layers to map back to output space
%
% Example:
%   net = tfno1d([16, 16, 16], 32, NumBlocks=4, InChannels=2, OutChannels=1);

% Copyright 2026 The MathWorks, Inc.

    arguments
        numModes (1, 3) double {mustBePositive, mustBeInteger, mustBeFinite} = [16, 16, 16]
        latentChannelSize (1, 1) {mustBePositive, mustBeInteger, mustBeFinite} = 16
        args.NumBlocks (1, 1) double {mustBePositive, mustBeInteger, mustBeFinite} = 2
        args.ExpandNet (1, 1) logical = true
        args.LiftingChannelRatio (1, 1) double = 2
        args.ProjectionChannelRatio (1, 1) double = 2
        args.InChannels (1, 1) double = 1
        args.OutChannels (1, 1) double = 1
        args.SpatialLimits (3, 2) double = [0 1; 0 1; 0 1]
        args.SpectralRank (1, 1) double {mustBeInRange(args.SpectralRank, 0, 1)} = 1
        args.LinearFNOSkip (1, 1) logical = true
        args.ChannelMLPSkip (1, 1) logical = true
        args.UseSpectralConvBias (1, 1) logical = true
        args.FuseSpectralConv (1, 1) logical = true
    end

    liftingChannels = args.LiftingChannelRatio * latentChannelSize;
    
    layers = [inputLayer([NaN latentChannelSize latentChannelSize latentChannelSize args.InChannels],"BSSSC", Name="input"), ...
        spatialEmbeddingLayer3D(args.SpatialLimits, Name="positionEmbdding"), ...
        depthConcatenationLayer(2, Name="concat"), ...
        convolution3dLayer(1, liftingChannels, Name="lifting1"), ...
        convolution3dLayer(1, latentChannelSize, Name="lifting2")];

    for i = 1:args.NumBlocks
        layers(end+1) = fnoBlock3D(numModes, latentChannelSize, ...
            Rank=args.SpectralRank, ...
            LinearFNOSkip=args.LinearFNOSkip, ...
            ChannelMLPSkip=args.ChannelMLPSkip, ...
            UseSpectralConvBias=args.UseSpectralConvBias, ...
            FuseSpectralConv=args.FuseSpectralConv);
    end

    projectionChannels = args.ProjectionChannelRatio * latentChannelSize;
    
    layers = [layers, ...
        convolution3dLayer(1, projectionChannels, Name="proj1"), ... 
        geluLayer(Name="gelu1"), ...
        convolution3dLayer(1, args.OutChannels, Name="proj2")];    
    net = dlnetwork;
    net = addLayers(net, layers);
    net = connectLayers(net,"input","concat/in2");
    
    if args.ExpandNet
        net = expandLayers(net);
    end
end