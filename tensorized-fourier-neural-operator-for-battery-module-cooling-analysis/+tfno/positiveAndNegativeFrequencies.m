function [pos,neg] = positiveAndNegativeFrequencies(N)
%POSITIVENEGATIVEFREQUENCIES - Indices for positive and negative FFT frequencies.
%  [pos,neg] = POSITIVENEGATIVEFREQUENCIES(N) returns the indices into the 
%  positive and negative frequencies of a Fourier transform as computed by fft 
%  on a real valued input with N entries.
%
%  Inputs:
%    N - Number of entries in the real-valued input signal
%
%  Outputs:
%    pos - Indices of positive frequencies (2:(floor(N/2)+1))
%    neg - Indices of negative frequencies (N:-1:(ceil(N/2)+1))
%
%  For example:
%    >> u = rand(N,1);
%    >> uhat = fft(u);
%  The entries of uhat are:
%    uhat(1) -- the 0 frequency
%    uhat(pos) -- the positive frequencies
%    uhat(neg) -- the negative frequencies
%
%  Note that the negative frequencies are indexed from the end of the array
%  because uhat(N) corresponds to the negative of the same frequency as
%  uhat(2). As such, pos(i) and neg(i) are associated frequencies.

% Copyright 2026 The MathWorks, Inc.

pos = 2:(floor(N/2)+1);
neg = N:-1:(ceil(N/2)+1);
end
