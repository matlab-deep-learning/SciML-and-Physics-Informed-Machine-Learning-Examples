classdef tnnWithBackward < deep.DifferentiableFunction
    % tnnWithBackward   TNN with explicit gradient computation

    %   Copyright 2025 The MathWorks, Inc.

    properties
        AdjacencyMatrix
        SampleTime
        TemperatureIndices
    end

    methods
        function this = tnnWithBackward(adjMat, sampleTime, tempIdcs)
            numOutputs = 2;
            numMemoryVals = 4;
            this@deep.DifferentiableFunction(numOutputs, ...
                SaveInputsForBackward=true, ...
                SaveOutputsForBackward=true, ...
                NumMemoryValues=numMemoryVals);
            this.AdjacencyMatrix = adjMat;
            this.SampleTime = sampleTime;
            this.TemperatureIndices = tempIdcs;
        end

        function [Y, h, powerLoss, tempDiffs, conducts, sizeVector] = forward(this, X, h, W1_cn, b1_cn, W1_pl, b1_pl, W2_pl, b2_pl, caps)
            % Get sizes.
            [B, T] = size(X,[2 3]);
            szH = size(h,1);
            szC = size(W1_cn,1);
            szP = size(W2_pl,1);
            szTemps = szH + numel(this.TemperatureIndices);
            sizeVector = [szH szTemps B T];

            % Preallocate output arrays.
            Y = [];
            conducts = zeros([szC B T], Like=X);
            powerLoss = zeros([szP B T], Like=X);
            tempDiffs = zeros([szH B T], Like=X);

            for tt = 1:T
                Xtt = X(:,:,tt);
                temps = [h; Xtt(this.TemperatureIndices,:)];
                subNNInput = [Xtt; h];

                % Conductance network forward pass.
                conducts(:,:,tt) = iConductanceNet(subNNInput, W1_cn, b1_cn);

                % Power loss network forward pass.
                powerLoss(:,:,tt) = iPLoss(subNNInput, W1_pl, b1_pl, W2_pl, b2_pl);

                % Calculate temperature differences.
                tmp = (repelem(temps,szH,1)-repmat(h,szTemps,1)).*conducts(this.AdjacencyMatrix,:,tt);
                aa = reshape(tmp,szH,szTemps,B);
                aa = sum(aa,2);
                tempDiffs(:,:,tt) = permute(aa, [1 3 2]); %safe-squeeze

                % Calculate output.
                h = h + this.SampleTime.*exp(caps) .* (tempDiffs(:,:,tt) + powerLoss(:,:,tt));

                % Clip output and concat.
                h = max(min(h, 5), -1); 
                Y = cat(3, Y, h);
            end
        end

        function [dX, dh0, dW1_cn, db1_cn, dW1_pl, db1_pl, dW2_pl, db2_pl, dcaps] = backward(this, dY, dh, computeGradients, X, h0, W1_cn, b1_cn, W1_pl, b1_pl, W2_pl, b2_pl, caps, Y, ~, powerLoss, tempDiffs, conducts, sizeVector)
            szX = size(X,1);
            szH = sizeVector(1);
            szTemps = sizeVector(2);
            B = sizeVector(3);
            T = sizeVector(4);
            xInd = 1:szX;
            hInd = (1+szX):(szX+szH);
            hTindTemps = 1:szH;

            % Initialize gradients.
            if computeGradients(1)
                dX = zeros(size(X), Like=X);
            else
                dX = [];
            end
            dW1_cn = zeros(size(W1_cn), Like=W1_cn);
            db1_cn = zeros(size(b1_cn), Like=b1_cn);
            dW1_pl = zeros(size(W1_pl), Like=W1_pl);
            db1_pl = zeros(size(b1_pl), Like=b1_pl);
            dW2_pl = zeros(size(W2_pl), Like=W2_pl);
            db2_pl = zeros(size(b2_pl), Like=b2_pl);
            dcaps = zeros(size(caps), Like=caps);
            dh0tt = dh;

            % Backpropagate through time.
            for tt = T:-1:1
                dhtt = dY(:, :, tt) + dh0tt;
                htt = Y(:, :, tt);
                if tt > 1
                    h0tt = Y(:, :, tt-1);
                else
                    h0tt = h0;
                end
                subNNInput = [X(:,:,tt); h0tt];
                temps = [h0tt; X(this.TemperatureIndices,:,tt)];

                % Gradient of caps.
                dhtt = iBackwardClip(dhtt, htt);
                dcaps_tt = iBackwardCaps(dhtt, this.SampleTime, caps, tempDiffs(:,:,tt), powerLoss(:,:,tt));

                % Power loss gradients.
                dPowerLoss = dhtt.*(this.SampleTime.*exp(caps));
                [dsubNNInput1, dW1_pl_tt, db1_pl_tt, dW2_pl_tt, db2_pl_tt] = iBackwardPowerLoss(dPowerLoss, subNNInput, W1_pl, b1_pl, W2_pl, b2_pl);

                % Temp diff gradients.
                dtempDiffs = dhtt.*(this.SampleTime.*exp(caps));
                dtempDiffs = permute(dtempDiffs, [1 3 2]);
                dtmp = repmat(dtempDiffs, [1 szTemps 1]);

                % Conductance net gradients.
                dtmp = reshape(dtmp, szH*szTemps, B);
                dcconductsAdjMat = (repelem(temps,szH,1)-repmat(h0tt,szTemps,1)).*dtmp;
                adjIdx = this.AdjacencyMatrix(:);
                dconducts = zeros(size(conducts, [1 2]));
                for ii = 1:numel(adjIdx)
                    dconducts(adjIdx(ii),:) = dconducts(adjIdx(ii), :) + dcconductsAdjMat(ii, :);
                end
                [dsubNNInput2, dW1_cn_tt, db1_cn_tt] = iBackwardConductanceNet(dconducts, subNNInput, W1_cn, b1_cn, conducts(:,:,tt));

                % State gradient.
                dsubNNInput = dsubNNInput1 + dsubNNInput2;
                dh01 = dsubNNInput(hInd, :);

                drepelemTemps = dtmp.*conducts(this.AdjacencyMatrix, :, tt);
                dtemps = zeros(size(temps), Like=temps);
                for jj = 1:szTemps
                    repelemIdx = (1+(jj-1)*szH):jj*szH;
                    dtemps(jj,:) = sum( drepelemTemps(repelemIdx,:), 1 );
                end
                dh02 = dtemps(hTindTemps,:);

                drepmatH0 = -dtmp.*conducts(this.AdjacencyMatrix, :, tt);
                dh03 = zeros(size(h0), Like=h0);
                for ll = 1:szH
                    repmatIdx = ll:szH:(szH*szTemps);
                    dh03(ll,:) = sum( drepmatH0(repmatIdx, :), 1 );
                end
                dh0tt = dh01 + dh02 + dh03 + dhtt;

                % Accumulate gradients through time.
                if computeGradients(1)
                    dX1 = dsubNNInput(xInd, :);
                    dX2_temp_idcs = dtemps(szH+1:end,:);
                    dX2 = zeros(size(dX1), Like=dX1);
                    for kk = 1:numel(this.TemperatureIndices)
                        dX2(this.TemperatureIndices(kk), :) = dX2_temp_idcs(kk,:);
                    end
                    dX(:,:,tt) = dX1 + dX2;
                end
                dW1_cn = dW1_cn + dW1_cn_tt;
                db1_cn = db1_cn + db1_cn_tt;
                dW1_pl = dW1_pl + dW1_pl_tt;
                db1_pl = db1_pl + db1_pl_tt;
                dW2_pl = dW2_pl + dW2_pl_tt;
                db2_pl = db2_pl + db2_pl_tt;
                dcaps = dcaps + dcaps_tt;
            end

            % Initial state gradient.
            dh0 = dh0tt;
        end

    end

end

function y = iConductanceNet(x, W1_cn, b1_cn)
y = W1_cn*x + b1_cn;
y = iSigmoid(y);
y = abs(y);
end

function x = iSigmoid(x)
x = 1./(1 + exp(-x));
end

function y = iPLoss(x, W1_pl, b1_pl, W2_pl, b2_pl)
y = W1_pl*x + b1_pl;
y = tanh(y);
y = W2_pl*y + b2_pl;
y = abs(y);
end

function dhtt = iBackwardClip(dhtt, htt)
idxGT = (htt >= -1);
idxLT = (htt <= 5);
mask = idxGT.*idxLT;
dhtt = mask.*dhtt;
end

function dcaps = iBackwardCaps(dhtt, sample_time, caps, temp_diffs, power_loss)
dcaps = dhtt.*(sample_time.*exp(caps).*(temp_diffs + power_loss) );
dcaps = sum(dcaps,2);
end

function [dx, dW1_pl, db1_pl, dW2_pl, db2_pl] = iBackwardPowerLoss(dy4, x, W1_pl, b1_pl, W2_pl, b2_pl)
y1= W1_pl*x + b1_pl;
y2 = tanh(y1);
y3 = W2_pl*y2 + b2_pl;

dy3 = iBackwardAbs(dy4, y3);

[dy2, dW2_pl, db2_pl] = iBackwardFC(dy3, y2, W2_pl);

dy1 = iBackwardTanh(dy2, y2);

[dx, dW1_pl, db1_pl] = iBackwardFC(dy1, x, W1_pl);
end

function dx = iBackwardAbs(dy, x)
posMask = x >= 0;
negMask = x < 0;

dx = dy.*(posMask + -1.*negMask);
end

function [dx, dw, db] = iBackwardFC(dy, x, w)
dw = dy*x';
db = sum(dy,2);
dx = w'*dy;
end

function dx = iBackwardTanh(dy,y)
dx = dy.*(1 - y.^2);
end

function [dx, dW1_cn, db1_cn] = iBackwardConductanceNet(dy2, x, W1_cn, ~, y2)
dy1 = iBackwardSigmoid(dy2, y2);
[dx, dW1_cn, db1_cn] = iBackwardFC(dy1, x, W1_cn);
end

function dx = iBackwardSigmoid(dy, y)
dx = dy.*y.*(1 - y);
end