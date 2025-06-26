function layer = fourierLayer(numModes,tWidth,args)

arguments
    numModes
    tWidth
    args.Name = ""
end
name = args.Name;

net = dlnetwork;

layers = [
    identityLayer(Name="in")
    spectralConvolution1dLayer(numModes,tWidth,Name="specConv")
    additionLayer(2,Name="add")];

net = addLayers(net,layers);

layer = convolution1dLayer(1,tWidth,Name="fc");
net = addLayers(net,layer);

net = connectLayers(net,"in","fc");
net = connectLayers(net,"fc","add/in2");

layer = networkLayer(net,Name=name);

end