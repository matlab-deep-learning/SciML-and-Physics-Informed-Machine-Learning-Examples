function net = transolver(args)
% transolver Construct a Transolver dlnetwork.
%
%   net = transolver.transolver() creates a Transolver dlnetwork with
%   default hyperparameters. The network is returned uninitialized.
%
%   Name-Value Arguments:
%       NumLayers          - Number of Transolver blocks. Default: 2
%       NumHeads           - Number of attention heads. Default: 1
%       HiddenSize         - Hidden feature dimension. Default: 64
%       NumSlices          - Number of physics slices. Default: 32
%       OutputSize         - Number of output channels. Default: 1
%       IncludeTemperature - Include learnable temperature scaling in
%                            attention. Default: false
%       Dropout            - Dropout probability. Default: 0
%       Rezero             - Use ReZero residual initialization. Default: false
%       NormalizationLayer - Handle to normalization layer constructor.
%                            Default: @layerNormalizationLayer
%
%   Output:
%       net - Uninitialized dlnetwork.

%   Copyright 2026 The MathWorks, Inc.

arguments
    args.NumLayers (1,1) double {mustBePositive,mustBeInteger} = 2
    args.NumHeads (1,1) double {mustBePositive,mustBeInteger} = 1
    args.HiddenSize (1,1) double {mustBePositive,mustBeInteger} = 64
    args.NumSlices (1,1) double {mustBePositive,mustBeInteger} = 32
    args.OutputSize (1,1) double {mustBePositive,mustBeInteger} = 1
    args.IncludeTemperature (1,1) logical = false
    args.Dropout (1,1) double = 0 
    args.Rezero (1,1) logical = false
    args.NormalizationLayer = @layerNormalizationLayer
end

transolvers = repmat(transolverBlock(args.NumHeads,args.HiddenSize,args.NumSlices,args.IncludeTemperature,args.Dropout,args.Rezero,args.NormalizationLayer), args.NumLayers, 1);
for i = 1:args.NumLayers
    transolvers(i).Name = "transolver_"+i;
end

layers = [
    convolution1dLayer(1,args.HiddenSize)
    transolvers
    args.NormalizationLayer()
    convolution1dLayer(1,args.OutputSize)];
net = dlnetwork(layers,Initialize = false);
net = addLayers(net,identityLayer(Name="mask"));
for i = 1:args.NumLayers
    net = connectLayers(net,"mask", "transolver_"+i+"/mask");
end
end

function block = transolverBlock(numHeads,hiddenSize,numSlices,temp,dropout,rezero,norm)
block = [
    identityLayer(Name="X")
    norm()
    transolver.transolverLayer(hiddenSize,numSlices,numHeads,Name="transolver",IncludeTemperature=temp)
    dropoutLayer(dropout)];

if rezero
    block = [block; rezeroLayer(hiddenSize)];
end

block = [block
    additionLayer(2,Name="add1")
    norm()
    convolution1dLayer(1,hiddenSize*4)
    geluLayer
    convolution1dLayer(1,hiddenSize)
    dropoutLayer(dropout)];

if rezero
    block = [block; rezeroLayer(hiddenSize)];
end
block = [block;additionLayer(2,Name="add2")];
block = dlnetwork(block,Initialize=false);
block = addLayers(block,identityLayer(Name="mask"));
block = connectLayers(block,"mask","transolver/mask");
block = connectLayers(block,"X","add1/in2");
block = connectLayers(block,"add1","add2/in2");
block = networkLayer(block);
end

function layer = rezeroLayer(hiddenSize)

layer = convolution1dLayer(1,hiddenSize,WeightsInitializer="zeros",BiasLearnRateFactor=0);
end