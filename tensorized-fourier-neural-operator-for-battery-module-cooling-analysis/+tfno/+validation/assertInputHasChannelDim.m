function assertInputHasChannelDim(N, numChannels)
    if isempty(numChannels)
        error("depthwiseConv" + N + "dLayer must have exactly 1 channel dimension");
    end
end

