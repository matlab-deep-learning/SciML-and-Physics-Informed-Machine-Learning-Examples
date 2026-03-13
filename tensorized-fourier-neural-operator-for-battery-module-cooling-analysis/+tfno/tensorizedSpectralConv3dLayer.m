classdef tensorizedSpectralConv3dLayer < nnet.layer.Layer ...
        & nnet.layer.Formattable ...
        & nnet.layer.Acceleratable %#codegen
%TENSORIZEDSPECTRALCONV3DLAYER - 3D Spectral Convolution Layer.
%   layer = TENSORIZEDSPECTRALCONV3DLAYER(outChannels, numModes)
%   creates a spectral convolution 3d layer. outChannels
%   specifies the number of channels in the layer output.
%   numModes specifies the number of modes which are combined
%   in Fourier space for each of the 3 spatial dimensions.
%
%   layer = tensorizedSpectralConv3dLayer(outChannels, numModes,
%   Name=Value) specifies additional options using one or more
%   name-value arguments:
%
%     Name      - Name for the layer. The default value is "".
%     
%     Weights   - Complex learnable array of size
%                 [inChannels, outChannels, numModes(1), numModes(2), numModes(3)]. 
%                 The default value is [].
%
%     Bias      - Real learnable array of size [1, 1, 1, outChannels].
%                 The default value is [].
%
%     Rank      - Ratio of stored parameter count to full
%                 parameter count. Default 1 for no
%                 compression. If less than 1, uses a Tucker
%                 decomposition to represent the full tensor.
%
%     Fuse      - If true, applies the spectral convolution 
%                 using a fused contraction path that operates 
%                 directly on the factorized (Tucker) 
%                 representation without reconstructing the 
%                 dense spectral weight tensor. This typically 
%                 reduces memory and improves latency. If false,
%                 reconstructs the full spectral weight tensor
%                 from the factors and multiplies with the
%                 input in Fourier space. This can be slower 
%                 and more memory intensive.

% Copyright 2026 The MathWorks, Inc.

    properties
        NumChannels
        OutputSize
        NumModes
        Rank
        UseBias
        Fuse
    end

    properties (Dependent)
        Tensorized
    end

    properties (Learnable)
        Weights
        Core
        Factor1
        Factor2
        Factor3
        Factor4
        Factor5
        Bias
    end

    methods
        function this = tensorizedSpectralConv3dLayer(outChannels,numModes,args)
            arguments
                outChannels (1,1) double
                numModes    (1,3) double
                args.Name {mustBeTextScalar} = "tensorizedSpectralConv3d"
                args.Weights = []
                args.Bias double = []
                args.Rank (1,1) double {mustBeInRange(args.Rank, 0, 1)} = 1
                args.UseBias (1, 1) logical = true
                args.Fuse (1, 1) logical = true
            end

            this.OutputSize = outChannels;
            this.NumModes = numModes;
            this.Name = args.Name;
            this.Weights = args.Weights;
            this.Rank = args.Rank;
            this.UseBias = args.UseBias;
            this.Fuse = args.Fuse;

            if ~isempty(args.Weights) && this.Tensorized
                error("Rank must be 1 when providing weights.");
            end
        end

        function b = get.Tensorized(this)
            b = this.Rank < 1;
        end

        function this = initialize(this, ndl)
            inChannels = ndl.Size( finddim(ndl,'C') );
            outChannels = this.OutputSize;
            numModes = this.NumModes;
            numModes(2:end) = numModes(2:end)*2 - 1;
            this.NumChannels = inChannels;

            if isempty(this.Weights) || isempty(this.Core)
                if this.Tensorized
                    this = this.initializeTucker(inChannels, outChannels, numModes);
                else
                    std = 1./(inChannels+outChannels);
                    this.Weights = std.*randn([inChannels outChannels numModes], "like", 1i);
                end
            end

            if isempty(this.Bias) && this.UseBias
                this.Bias = zeros(1, 1, 1, this.OutputSize);
            end
        end

        function this = initializeTucker(this, inChannels, outChannels, numModes)
            sz = [inChannels outChannels numModes];
            contractFactor = this.Rank^(1/nnz(sz~=1));
            newSz = max(round(sz.*contractFactor), 1);

            % Core is an N-D tensor that is smaller than full tensor size of 
            % (inChannels)x(outChannels)x(numModes(1))x(numModes(2)).
            % There is 1 factor matrix per dimension in the core mapping
            % from the full tensor size to the core size.

            this.Core = 1./(inChannels+outChannels).*rand(newSz, 'like', 1i);

            this.Factor1 = 1./(newSz(1)+sz(1)).*rand([sz(1), newSz(1)], 'like', 1i);
            this.Factor2 = 1./(newSz(2)+sz(2)).*rand([sz(2), newSz(2)], 'like', 1i);
            this.Factor3 = 1./(newSz(3)+sz(3)).*rand([sz(3), newSz(3)], 'like', 1i);
            this.Factor4 = 1./(newSz(4)+sz(4)).*rand([sz(4), newSz(4)], 'like', 1i);
            this.Factor5 = 1./(newSz(5)+sz(5)).*rand([sz(5), newSz(5)], 'like', 1i);
        end

        function Xout = predict(this, X)
            N1 = size(X, 1);
            N2 = size(X, 2);
            N3 = size(X, 3);
            channelSize = size(X, 4);
            batchSize = size(X, 5);
            assert(channelSize == this.NumChannels);
            Nm1 = this.NumModes(1);
            Nm2 = this.NumModes(2);
            Nm3 = this.NumModes(3);
            d = dims(X);

            % Reshape into ConvDim1 x ConvDim2 x ConvDim3 x Channel x BatchedDims
            X = reshape(X,N1,N2,N3,channelSize,[]);

            % Compute FFT. Second and third dimensions may have complex values.
            X = real(X);
            X = stripdims(X);

            X = fft(fft(fft(X, [], 3), [], 2), [], 1);

            % Truncate to NumModes frequencies. In the first dimension we
            % don't require the negative frequencies due to conjugate
            % symmetry.
            xFreq = 1:Nm1;
            yPos = 1:Nm2;
            yNeg = (N2-Nm2+2):N2;
            yFreq = union(yPos,yNeg);
            zPos = 1:Nm3;
            zNeg = (N3-Nm3+2):N3;
            zFreq = union(zPos,zNeg);
            X = X(xFreq,yFreq,zFreq,:,:);

            % Permute the channel dimension to the front.
            X = permute(X,[4,5,1,2,3]);

            if this.Tensorized && this.Fuse
                % Directly contract X with the Tucker decomposition.
                X = tfno.tuckerContract(X, this.Core, {this.Factor1, ...
                    this.Factor2, ...
                    this.Factor3, ...
                    this.Factor4, ...
                    this.Factor5});
            else
                % Use dense weights, reconstructing if needed.
                if this.Tensorized
                    % Reconstruct dense tensor from Tucker decomposition.
                    weights = tfno.fullTensor(this.Core, {this.Factor1, ...
                        this.Factor2, ...
                        this.Factor3, ...
                        this.Factor4, ...
                        this.Factor5});
                else
                    weights = this.Weights;
                end
    
                % Potentially the weights can have too many modes in the 2nd
                % and 3rd dimension when NumModes was chose maximally and the
                % input convolution size is even. 
                weights = weights(:,:,:,1:min(size(X,4),size(weights,4)),1:min(size(X,5),size(weights,5)));
    
                % Perform the linear operation and permute back.
                X = pagemtimes(weights,X);
                X = permute(X,[3,4,5,1,2]);
            end

            % Zero out the unrequired frequencies
            Xout = zeros([N1,N2,N3,this.OutputSize,size(X,5)],like=X);
            Xout(xFreq,yFreq,zFreq,:,:) = X;

            % Make Xout conjugate symmetric so the inverse fft is
            % real-valued. See 
            % https://www.mathworks.com/help/matlab/ref/ifftn.html#bvjbzad-9

            % Xout(1,:,:,:,:) has the 2d conjugate symmetry
            [xPos,xNeg] = tfno.positiveAndNegativeFrequencies(N1);
            [yPos,yNeg] = tfno.positiveAndNegativeFrequencies(N2);
            [zPos,zNeg] = tfno.positiveAndNegativeFrequencies(N3);

            Xout(1,1,zNeg,:,:) = ...
                conj(Xout(1,1,zPos,:,:));
            Xout(1,yNeg,1,:,:) = ...
                conj(Xout(1,yPos,1,:,:));
            Xout(1,yNeg, zNeg,:,:) = ...
                conj(Xout(1,yPos,zPos,:,:));
            Xout(1,yNeg, zPos,:,:) = ...
                conj(Xout(1,yPos,zNeg,:,:));
            Xout(xNeg,1,1,:,:) = ...
                conj(Xout(xPos,1,1,:,:));

            % Xout(:,1,:,:,:) also has 2d conjugate symmetry
            Xout(xNeg,1,zNeg,:,:) = ...
                conj(Xout(xPos,1,zPos,:,:));
            Xout(xNeg,1,zPos,:,:) = ...
                conj(Xout(xPos,1,zNeg,:,:));

            % Xout(:,:,1,:,:) has the 2d conjugate symmetry
            Xout(xNeg,yNeg,1,:,:) = ...
                conj(Xout(xPos,yPos,1,:,:));
            Xout(xNeg,yPos,1,:,:) = ...
                conj(Xout(xPos,yNeg,1,:,:));

            % Xout(2:end,:,:,:,:) has 3d conjugate symmetry
            Xout(xNeg,yNeg,zNeg,:,:) = ...
                conj(Xout(xPos,yPos,zPos,:,:));
            Xout(xNeg,yPos,zNeg,:,:) = ...
                conj(Xout(xPos,yNeg,zPos,:,:));
            Xout(xNeg,yNeg,zPos,:,:) = ...
                conj(Xout(xPos,yPos,zNeg,:,:));
            Xout(xNeg,yPos,zPos,:,:) = ...
                conj(Xout(xPos,yNeg,zNeg,:,:));
            
            Xout = ifft(Xout,[],3);
            Xout = ifft(Xout,[],2);
            Xout = ifft(Xout,[],1,'symmetric');

            % Reshape back to original size
            Xout = reshape(Xout,[N1, N2, N3, this.OutputSize,batchSize]);
            
            if this.UseBias
                Xout = Xout + this.Bias;
            end

            Xout = dlarray(Xout,d);
            Xout = real(Xout);
        end
    end  
end

