function [loss, errL2, gtL2] = relativeL2Loss(pred, gt, params)
% RELATIVEL2LOSS - Compute the relative L2 loss between predictions and ground truth.
%   LOSS = RELATIVEL2LOSS(PRED, GT) computes the relative L2 loss of the 
%   predicted values PRED against ground truth GT with default parameters.
%
%   [LOSS, ERRL2, GTL2] = RELATIVEL2LOSS(PRED, GT, Name=Value) specifies 
%   additional options using one or more name-value arguments:
%
%     Reduction  - Method for reducing the loss across batch.
%                  Options are 'mean', 'sum', or 'none'.
%                  The default value is 'mean'.
%
%     SquareRoot - If true, takes the square root of L2 norm.
%                  If false, uses squared L2 norm.
%                  The default value is false.
%
%     Normalize  - If true, normalizes the L2 norm.
%                  The default value is false.
%
%     Epsilon      - Small constant to add to denominator to avoid division
%                    by zero, in single precision.
%                    The default value is 2e-16.
%
%   The relative L2 loss is defined as:
%     loss = ||pred - gt||_{L^2} / ||gt||_{L^2}
%   which is calculated per sample in the batch and then reduced.
%
%   This loss function is useful for problems where the scale of the 
%   target values varies significantly.
%
%   Inputs PRED and GT must be dlarrays of identical size.
%
%   Outputs:
%     LOSS  - Relative L2 loss value
%     ERRL2 - L2 norm of the error (gt - pred)  
%     GTL2  - L2 norm of the ground truth
%
%   Example:
%     pred = dlarray(randn(10, 5));
%     gt = dlarray(randn(10, 5));
%     loss = relativeL2Loss(pred, gt);

% Copyright 2026 The MathWorks, Inc.

    arguments
        pred dlarray
        gt dlarray
        params.Reduction (1,1) string {mustBeMember(params.Reduction, {'mean', 'sum', 'none'})} = "mean"
        params.SquareRoot (1,1) logical = false
        params.Normalize (1,1) logical = false
        params.Epsilon (1, 1) single = 2e-16
    end

    pred = lossFunctions.permuteDimFirst(pred, "B");
    gt = lossFunctions.permuteDimFirst(gt, "B");

    if ~isequal(size(pred), size(gt))
        error('pred and gt must have identical size.');
    end

    err = gt - pred;

    errL2 = lossFunctions.l2Norm(err, ...
        Normalize=params.Normalize, ...
        Reduction='none', ...
        SquareRoot=params.SquareRoot);
    gtL2 = lossFunctions.l2Norm(gt, ...
        Normalize=params.Normalize, ...
        Reduction='none', ...
        SquareRoot=params.SquareRoot);

    loss = errL2./(gtL2 + params.Epsilon);
   
     switch params.Reduction
         case "mean"
             loss = mean(loss);
         case "sum"
             loss = sum(loss);
     end
end