function layer = fnoBlock3D(numModes,latentChannelSize,args)
%FNOBLOCK3D - Create a 3D Fourier Neural Operator (FNO) block.
%  layer = FNOBLOCK3D(numModes, latentChannelSize) creates a 3D FNO block
%  that combines spectral convolution in Fourier space with channel-wise
%  MLPs for feature transformation. The block includes optional skip 
%  connections and normalization layers. It is returned as a networkLayer.
%
%  Inputs:
%    numModes - Number of Fourier modes to use in spectral convolution
%    latentChannelSize - Number of channels in the latent representation
%
%  Supported Name-Value pairs are:
%    "NumMLPLayers"            Number of MLP layers (default: 2)
%    "MLPExpansion"            Channel expansion factor for MLP (default: 0.5)
%    "Name"                    Name for the layer (default: "")
%    "LinearFNOSkip"           Enable linear skip connection for FNO (default: true)
%    "ChannelMLPSkip"          Enable channel-wise skip connection (default: true)
%    "Rank"                    Rank for spectral convolution (default: 1)
%    "UseSpectralConvBias"     Use bias in spectral convolution (default: true)
%    "FuseSpectralConv"       Directly contract spectral activations with tensorized weights (default: true)
%
%  The block consists of:
%    1. Spectral convolution in Fourier domain
%    2. Layer normalization and residual connection
%    3. Channel-wise MLP with expansion and contraction
%    4. Optional skip connections for improved gradient flow

% Copyright 2026 The MathWorks, Inc.

arguments
    numModes (1, 3) double {mustBePositive, mustBeInteger, mustBeFinite}
    latentChannelSize (1, 1) double {mustBePositive, mustBeInteger, mustBeFinite}
    args.NumMLPLayers (1, 1) double {mustBePositive, mustBeInteger, mustBeFinite} = 2
    args.MLPExpansion (1, 1) double {mustBePositive, mustBeLessThan(args.MLPExpansion, 1)} = 0.5
    args.Name (1, 1) string = ""
    args.LinearFNOSkip (1, 1) logical = true
    args.ChannelMLPSkip (1, 1) logical = true
    args.Rank (1, 1) double {mustBeInRange(args.Rank, 0, 1)} = 1
    args.UseSpectralConvBias (1, 1) logical = true
    args.FuseSpectralConv (1, 1) logical = true
end

    name = args.Name;
    
    net = dlnetwork;
    
    layers = [identityLayer(Name="in"), ...
        tensorizedSpectralConv3dLayer(...
            latentChannelSize, ...
            numModes, ...
            Rank=args.Rank, ...
            UseBias=args.UseSpectralConvBias, ...
            Name="specConv", ...
            Fuse=args.FuseSpectralConv), ...
        layerNormalizationLayer(Name="ln1"), ...
        additionLayer(2, Name="add1"), ...
        geluLayer(Name="gelu1"), ...
        convolution3dLayer(1, latentChannelSize * args.MLPExpansion, Name="channelMLP1"), ...
        geluLayer(Name="gelu2"), ...
        convolution3dLayer(1, latentChannelSize, Name="channelMLP2"), ...
        layerNormalizationLayer(Name="ln2"), ...
        additionLayer(2, Name="add2"), ...
        geluLayer(Name="gelu3")];
    
    net = addLayers(net, layers);

    if args.LinearFNOSkip
        net = addLayers(net, convolution3dLayer(1, latentChannelSize, Name="fnoSkip", BiasLearnRateFactor=0));
        net = connectLayers(net, "in", "fnoSkip");
        net = connectLayers(net, "fnoSkip", "add1/in2");
    else
        net = connectLayers(net, "in", "add1/in2");
    end

    if args.ChannelMLPSkip
        net = addLayers(net, depthwiseConv3dLayer(latentChannelSize, Name="channelSkip"));
        net = connectLayers(net, "in", "channelSkip");
        net = connectLayers(net, "channelSkip", "add2/in2");
    else
        net = connectLayers(net, "in", "add2/in2");
    end

    layer = networkLayer(net,Name=name);
end