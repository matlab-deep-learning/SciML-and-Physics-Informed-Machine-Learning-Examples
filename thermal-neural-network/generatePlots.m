function generatePlots(profileLengths,test_tensor,prediction,scale,targetNames)
% Visualize performance of the trained TNN on the test data set by comparing
% the predictions to the ground truth

%   Copyright 2025 The MathWorks, Inc.

f = figure;
numOutputs = size(prediction,1);
numInputs = size(test_tensor,1)-numOutputs;
tiledlayout(size(test_tensor,2),numOutputs)
for ii = 1:size(test_tensor,2)
    for jj = 1:numOutputs
        nexttile
        groundTruth= scale*squeeze(extractdata(test_tensor(numInputs+jj,ii,1:profileLengths(ii))));
        currentPred = scale*squeeze(prediction(jj,ii,1:profileLengths(ii)));
        maxError = max(abs(groundTruth-currentPred));
        mseError = mse(groundTruth,currentPred);
        plot(0:profileLengths(ii)-1,groundTruth, ...
            Color = "green",LineWidth = 1.5)
        hold on
        plot(0:profileLengths(ii)-1,currentPred, ...
            Color= "blue",LineWidth = 1.5)
        textContent = sprintf('MSE: %1.1f K^2\nmax.abs.: %1.1f K',mseError,maxError);
        text(0.5,0.8,textContent,Units = "normalized",Interpreter = "none",FontSize = 7,Color = 'r')
        if ii == 1
            title(targetNames{jj},Interpreter="none")
            if jj==1
                legend(["Ground truth","Prediction MATLAB"],Location = "southeast")
            end

        end
        if jj==1
            ylabel("Profile "+ii+newline+"Temp. in °C")
        end

    end
end
