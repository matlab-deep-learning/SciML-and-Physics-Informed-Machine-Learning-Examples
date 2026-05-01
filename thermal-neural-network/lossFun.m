function [weightedLoss,gradients,hidden] = lossFun(model, input, target, hidden, sampleWeights)
% lossFun   Weighted MSE loss

%   Copyright 2025 The MathWorks, Inc.

% Forward pass
[output, hidden] = predict(model, input, hidden);

% Compute loss
loss = l2loss(output, target, Reduction="none");

% Sample weighting
weightedLoss = sum((loss .* sampleWeights) ./ sum(sampleWeights, 'all'),'all');

% Gradient calculation
gradients = dlgradient(weightedLoss, model.Learnables, EnableHigherDerivatives=false);
end