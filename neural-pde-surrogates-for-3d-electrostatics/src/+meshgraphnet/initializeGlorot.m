function W = initializeGlorot(nin,nout)
    % initializeGlorot Initialize a weight matrix using Glorot uniform initialization.
    %
    %   W = meshgraphnet.initializeGlorot(nin, nout) returns a weight matrix
    %   of size nout-by-nin sampled uniformly from [-s, s] where
    %   s = sqrt(6 / (nin + nout)).
    %
    %   Inputs:
    %       nin  - Number of input units (fan-in).
    %       nout - Number of output units (fan-out).
    %
    %   Output:
    %       W - Weight matrix of size nout-by-nin.

    %   Copyright 2026 The MathWorks, Inc.
    scale = sqrt(6/(nin+nout));
    W = 2*rand(nout,nin) - 1;
    W = scale*W;
end