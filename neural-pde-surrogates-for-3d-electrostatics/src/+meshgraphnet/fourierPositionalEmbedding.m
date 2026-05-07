function x = fourierPositionalEmbedding(x,maxfreq)
    % fourierPositionalEmbedding Augment features with Fourier positional embeddings.
    %
    %   x = meshgraphnet.fourierPositionalEmbedding(x, maxfreq) appends
    %   cos(f*pi*x) and sin(f*pi*x) for frequencies f = 1:maxfreq to the
    %   input features. The output has size (2*maxfreq+1)*C-by-N.
    %
    %   Inputs:
    %       x       - Feature matrix of size C-by-N with values in [-1,1].
    %       maxfreq - Maximum frequency for the Fourier embedding.
    %
    %   Output:
    %       x - Augmented feature matrix of size C*(2*maxfreq+1)-by-N.

    %   Copyright 2026 The MathWorks, Inc.
    f = x*pi;
    F = permute(1:maxfreq,[1,3,2]);
    f = f.*F;
    f = permute(f,[1,3,2]);
    f = reshape(f,[],size(x,2));
    x = cat(1,x,cos(f),sin(f));
end