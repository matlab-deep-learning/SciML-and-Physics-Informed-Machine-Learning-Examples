function assertValidNumConvolutionDimensions(N, hasTimeDimension, numSpatialDimensions)
    if ~(hasTimeDimension + numSpatialDimensions == N)
        error("The total number of spatial and time dimensions in " + ...
            "depthwiseConv" + N + "dLayer must be " + N + ".");
    end
end
