function layer = transolverLayer(hiddenSize,numSlices,numHeads,args)
% transolverLayer Create a Transolver physics-attention layer as a networkLayer.
%
%   layer = transolver.transolverLayer(hiddenSize, numSlices, numHeads)
%   constructs a single Transolver attention block that computes slice 
%   representations, applies self-attention across slices, and maps back 
%   to per-node features.
%
%   Inputs:
%       hiddenSize - Hidden feature dimension.
%       numSlices  - Number of physics slices.
%       numHeads   - Number of attention heads.
%
%   Name-Value Arguments:
%       Name               - Layer name. Default: ""
%       IncludeTemperature - Include learnable temperature scaling.
%                            Default: false
%
%   Output:
%       layer - A networkLayer wrapping the Transolver attention block.

%   Copyright 2026 The MathWorks, Inc.

arguments
    hiddenSize (1,1) {mustBePositive,mustBeInteger}
    numSlices (1,1) {mustBePositive,mustBeInteger}
    numHeads (1,1) {mustBePositive,mustBeInteger}
    args.Name (1,1) string = ""
    args.IncludeTemperature (1,1) logical = false
end


layers = [
    identityLayer(Name="X")
    convolution1dLayer(1,numSlices)];
if args.IncludeTemperature
    layers = [layers
        functionLayer(@(x,temp) x./max(temp,0.01), InputNames=["x","temp"],Acceleratable=true,Name="temp")];
end
layers = [layers       
    softmaxLayer(Name="sliceWeights")
    sliceRepresentationLayer(Name="sliceRepresentation")
    selfAttentionLayer(numHeads,hiddenSize)
    nodeRepresentationLayer(Name="nodeRepresentation")
    convolution1dLayer(1,hiddenSize)];

net = dlnetwork(layers,Initialize=false);
net = addLayers(net,identityLayer(Name="mask"));
net = connectLayers(net,"mask","sliceRepresentation/mask");
net = connectLayers(net,"X","sliceRepresentation/X");
net = connectLayers(net,"sliceWeights","nodeRepresentation/W");

if args.IncludeTemperature
    % see proj_temperature here: https://github.com/thuml/Transolver_plus/blob/main/models/Transolver_plus.py
    net = addLayers(net,[convolution1dLayer(1,numSlices,Name="tempProjectionIn"); geluLayer; convolution1dLayer(1,1); geluLayer(); convolution1dLayer(1,1,WeightLearnRateFactor=0,Weights=1,Bias=0.5,Name="tempProjection")]);    
    net = connectLayers(net,"X","tempProjectionIn");
    net = connectLayers(net,"tempProjection","temp/temp");
end
layer = networkLayer(net,Name=args.Name);
end

function layer = sliceRepresentationLayer(args)
arguments
    args.Name (1,1) string = ""
end
layer = functionLayer(@sliceLinear,...
    InputNames=["W","X","mask"],...
    Formattable=true,...
    Acceleratable=true,...
    Name=args.Name);
end

function Z = sliceLinear(W,X,mask)
W = W.*mask;
W = stripdims(W,"CSB");
Z = pagemtimes(W,stripdims(X));
Z = Z ./ (sum(W,2)+1e-6);
Z = dlarray(Z,"SCB");
end

function layer = nodeRepresentationLayer(args)
arguments
    args.Name (1,1) string = ""
end
layer = functionLayer(@nodeLinear,...
    Formattable=true,...
    Acceleratable=true,...
    InputNames=["Z","W"],...
    Name=args.Name);
end

function Z = nodeLinear(Z,W)
Z = pagemtimes(stripdims(W), stripdims(Z));
Z = dlarray(Z,"SCB");
end
