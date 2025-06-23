classdef spectralConvolution1dLayer < nnet.layer.Layer ...
        & nnet.layer.Formattable ...
        & nnet.layer.Acceleratable
    % spectralConvolution1dLayer   1-D Spectral Convolution Layer

    properties
        NumChannels
        OutputSize
        NumModes
    end
    
    properties (Learnable)
        Weights
    end
    
    methods
        function this = spectralConvolution1dLayer(outChannels,numModes,args)
            % spectralConvolution1dLayer   1-D Spectral Convolution Layer
            %
            %   layer = spectralConvolution1dLayer(outChannels, numModes)
            %   creates a spectral convolution 1d layer. outChannels
            %   specifies the number of channels in the layer output.
            %   numModes specifies the number of modes which are combined
            %   in Fourier space.
            %
            %   layer = spectralConvolution1dLayer(outChannels, numModes,
            %   Name=Value) specifies additional options using one or more
            %   name-value arguments:
            %
            %     Name      - Name for the layer. The default value is "".
            %     
            %     Weights   - Complex learnable array of size
            %                 (inChannels)x(outChannels)x(numModes). The
            %                 default value is [].

            arguments
                outChannels (1,1) double
                numModes    (1,1) double
                args.Name {mustBeTextScalar} = "spectralConv1d"
                args.Weights = []
            end
            
            this.OutputSize = outChannels;
            this.NumModes = numModes;
            this.Name = args.Name;
            this.Weights = args.Weights;
        end

        function this = initialize(this, ndl)
            inChannels = ndl.Size( finddim(ndl,'C') );
            outChannels = this.OutputSize;
            numModes = this.NumModes;

            if isempty(this.Weights)
                this.NumChannels = inChannels;
                this.Weights = 1./(inChannels*outChannels).*( ...
                    rand([inChannels outChannels numModes]) + ...
                    1i.*rand([inChannels outChannels numModes]) );
            else
                assert( inChannels == this.NumChannels, 'The input channel size must match the layer' );
            end
        end
        
        function y = predict(this, x)
            % First compute the rfft, normalized and one-sided
            x = real(x);
            x = stripdims(x);
            N = size(x, 1);
            xft = iRFFT(x, 1, N);
            
            % Multiply selected Fourier modes
            xft = permute(xft(1:this.NumModes, :, :), [3 2 1]);
            yft = pagemtimes( xft, this.Weights );
            yft = permute(yft, [3 2 1]);

            S = floor(N/2)+1 - this.NumModes;
            yft = cat(1, yft, zeros([S size(yft, 2:3)], 'like', yft));
            
            % Return to physical space via irfft, normalized and one-sided
            y = iIRFFT(yft, 1, N);
            
            % Re-apply labels
            y = dlarray(y, 'SCB');
            y = real(y);
        end
    end  
end

function y = iRFFT(x, dim, N)
y = fft(x, [], dim); 
y = y(1:floor(N/2)+1, :, :)./N;
end

function y = iIRFFT(x, dim, N)
x(end+1:N, :, :, :) = conj( x(ceil(N/2):-1:2, :, :, :) );
y = ifft(N.*x, [], dim, 'symmetric');
end
