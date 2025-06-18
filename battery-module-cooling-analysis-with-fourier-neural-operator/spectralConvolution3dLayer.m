classdef spectralConvolution3dLayer < nnet.layer.Layer ...
        & nnet.layer.Formattable ...
        & nnet.layer.Acceleratable
    % spectralConvolution3dLayer   A custom layer implementation of
    % spectral convolution for data with 3 spatial dimensions.
    
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

            if isempty(this.Weights)
                this.Cin = inChannels;
                this.Weights = 1./(inChannels*outChannels).*( ...
                    rand([outChannels inChannels numModes numModes numModes]) + ...
                    1i.*rand([outChannels inChannels numModes numModes numModes]) );
            end
        end
        
        function y = predict(this, x)
            
            % Compute the 3d fft and retain only the low frequency modes as
            % specified by NumModes.
            x = real(x);
            x = stripdims(x);
            N = size(x, 1);
            Nm = this.NumModes;
            xft = fft(x, [], 1);
            xft = xft(1:Nm,:,:,:,:);
            xft = fft(xft, [], 2);
            xft = xft(:,1:Nm,:,:,:);
            xft = fft(xft, [], 3);
            xft = xft(:,:,1:Nm,:,:);
            
            % Multiply selected Fourier modes with the learnable weights.
            xft = permute(xft, [4 5 1 2 3]);
            yft = pagemtimes( this.Weights, xft );
            yft = permute(yft, [3, 4, 5, 1, 2]);

            % Make the frequency representation conjugate-symmetric such
            % that the inverse Fourier transform is real-valued.
            S = floor(N/2)+1 - this.NumModes;
            idx = ceil(N/2):-1:2;
            yft = cat(1, yft, zeros([S size(yft, 2:5)], 'like', yft));
            yft = cat(1, yft, conj(yft(idx,:,:,:,:)));

            yft = cat(2, yft, zeros([size(yft,1), S, size(yft,3:5)], like=yft));
            yft = cat(2, yft, conj(yft(:,idx,:,:,:)));

            yft = cat(3, yft, zeros([size(yft,[1,2]), S, size(yft,4:5)], like=yft));
            yft = cat(3, yft, conj(yft(:,:,idx,:,:)));
            
            % Return to physical space via 3d ifft
            y = ifft(yft, [], 3, 'symmetric');
            y = ifft(y,[],2, 'symmetric');
            y = ifft(y,[],1, 'symmetric');
            
            % Re-apply labels
            y = dlarray(y, 'SSSCB');
            y = real(y);
        end
    end  
end