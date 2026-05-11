classdef spectralConvolution3dLayer < nnet.layer.Layer ...
        & nnet.layer.Formattable ...
        & nnet.layer.Acceleratable
    % spectralConvolution3dLayer   A custom layer implementation of
    % spectral convolution for data with 3 spatial dimensions.

    % Copyright 2025 The MathWorks, Inc.
    
    properties
        Cin
        Cout
        NumModes
    end
    
    properties (Learnable)
        Weights
    end
    
    methods
        function this = spectralConvolution3dLayer(numModes, outChannels, nvargs)
            arguments
                numModes    (1,1) double
                outChannels (1,1) double
                nvargs.Name {mustBeTextScalar} = "spectralConv3d"
                nvargs.Weights = []
            end
            
            this.Cout = outChannels;
            this.NumModes = numModes;
            this.Name = nvargs.Name;
            this.Weights = nvargs.Weights;
        end

        function this = initialize(this, ndl)
            inChannels = ndl.Size( finddim(ndl,'C') );
            outChannels = this.Cout;
            numModes = this.NumModes;
            M = 2*numModes - 1;

            if isempty(this.Weights)
                this.Cin = inChannels;
                this.Weights = 1./(inChannels*outChannels).*( ...
                    rand([outChannels inChannels M M numModes]) + ...
                    1i.*rand([outChannels inChannels M M numModes]) );
            end
        end

        function y = predict(this, x)
            x = real(x);
            x = stripdims(x);
            [N1,N2,N3] = size(x,[1,2,3]);
            Nm = this.NumModes;

            x = fft(x,[],1);
            x = fft(x,[],2);
            x = fft(x,[],3);

            % Retain the low frequency modes: DC, positive, and negative
            % frequencies in dims 1 & 2; only non-negative in dim 3.
            xFreq = union(1:Nm, N1-Nm+2:N1);
            yFreq = union(1:Nm, N2-Nm+2:N2);
            zFreq = 1:Nm;
            x = x(xFreq, yFreq, zFreq, :, :);

            % Multiply retained modes by learned weights.
            x = permute(x, [4 5 1 2 3]);
            W = this.Weights;
            W = W(:,:,1:min(size(x,3),size(W,3)),1:min(size(x,4),size(W,4)),:);
            x = pagemtimes(W, x);
            x = permute(x, [3 4 5 1 2]);

            % Place into full-size frequency grid.
            y = zeros([N1, N2, N3, size(x,4), size(x,5)], 'like', x);
            y(xFreq, yFreq, zFreq, :, :) = x;

            % Enforce conjugate symmetry so that the ifft is real-valued.
            [xPos,xNeg] = iPositiveAndNegativeFrequencies(N1);
            [yPos,yNeg] = iPositiveAndNegativeFrequencies(N2);
            [zPos,zNeg] = iPositiveAndNegativeFrequencies(N3);

            % 2d symmetry on the k3=0 plane
            y(xNeg,1,1,:,:) = conj(y(xPos,1,1,:,:));
            y(1,yNeg,1,:,:) = conj(y(1,yPos,1,:,:));
            y(xNeg,yNeg,1,:,:) = conj(y(xPos,yPos,1,:,:));
            y(xPos,yNeg,1,:,:) = conj(y(xNeg,yPos,1,:,:));

            % 1d symmetry on the k1=0,k2=0 line
            y(1,1,zNeg,:,:) = conj(y(1,1,zPos,:,:));

            % 2d symmetry on the k1=0 plane
            y(1,yNeg,zNeg,:,:) = conj(y(1,yPos,zPos,:,:));
            y(1,yPos,zNeg,:,:) = conj(y(1,yNeg,zPos,:,:));

            % 2d symmetry on the k2=0 plane
            y(xNeg,1,zNeg,:,:) = conj(y(xPos,1,zPos,:,:));
            y(xPos,1,zNeg,:,:) = conj(y(xNeg,1,zPos,:,:));

            % 3d symmetry for the interior octants
            y(xNeg,yNeg,zNeg,:,:) = conj(y(xPos,yPos,zPos,:,:));
            y(xPos,yNeg,zNeg,:,:) = conj(y(xNeg,yPos,zPos,:,:));
            y(xNeg,yPos,zNeg,:,:) = conj(y(xPos,yNeg,zPos,:,:));
            y(xPos,yPos,zNeg,:,:) = conj(y(xNeg,yNeg,zPos,:,:));

            % DC and Nyquist frequencies must be real.
            y(1,1,1,:,:) = real(y(1,1,1,:,:));
            if mod(N1,2)==0
                y(N1/2+1,1,1,:,:) = real(y(N1/2+1,1,1,:,:));
            end
            if mod(N2,2)==0
                y(1,N2/2+1,1,:,:) = real(y(1,N2/2+1,1,:,:));
            end
            if mod(N3,2)==0
                y(1,1,N3/2+1,:,:) = real(y(1,1,N3/2+1,:,:));
            end
            if mod(N1,2)==0 && mod(N2,2)==0
                y(N1/2+1,N2/2+1,1,:,:) = real(y(N1/2+1,N2/2+1,1,:,:));
            end
            if mod(N1,2)==0 && mod(N3,2)==0
                y(N1/2+1,1,N3/2+1,:,:) = real(y(N1/2+1,1,N3/2+1,:,:));
            end
            if mod(N2,2)==0 && mod(N3,2)==0
                y(1,N2/2+1,N3/2+1,:,:) = real(y(1,N2/2+1,N3/2+1,:,:));
            end
            if mod(N1,2)==0 && mod(N2,2)==0 && mod(N3,2)==0
                y(N1/2+1,N2/2+1,N3/2+1,:,:) = real(y(N1/2+1,N2/2+1,N3/2+1,:,:));
            end

            % Return to physical space.
            y = ifft(y,[],3);
            y = ifft(y,[],2);
            y = ifft(y,[],1,'symmetric');
            y = dlarray(y, 'SSSCB');
        end
    end
end

function [pos,neg] = iPositiveAndNegativeFrequencies(N)
pos = 2:(floor(N/2)+1);
neg = N:-1:(ceil(N/2)+1);
end