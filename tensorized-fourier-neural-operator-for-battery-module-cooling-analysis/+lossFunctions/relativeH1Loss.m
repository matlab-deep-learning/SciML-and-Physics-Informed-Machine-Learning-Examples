function loss = relativeH1Loss(pred, gt, params)
%RELATIVEH1LOSS - Compute the relative H1 norm loss between predictions and ground truth.
%   LOSS = RELATIVEH1LOSS(PRED, GT) computes the relative H1 norm loss
%   between predicted values PRED and ground truth values GT with default
%   parameters.
%
%   LOSS = RELATIVEH1LOSS(PRED, GT, Name=Value) specifies additional options
%   using one or more name-value arguments:
%
%     Normalize    - If true, normalizes the H1 norm.
%                    The default value is false.
%
%     SpatialSizes - 1xD vector of physical domain sizes for each spatial
%                    dimension. The default value is ones(1,D).
%
%     SquareRoot   - If true, returns the square root of the norm.
%                    If false, returns the squared norm.
%                    The default value is false.
%
%     Reduction    - Method for reducing the loss across batch.
%                    Options are 'mean', 'sum', or 'none'.
%                    The default value is 'mean'.
%
%     Periodic     - 1xD logical array indicating which spatial
%                    dimensions are periodic. The default value
%                    is true for all dimensions.
%
%     Epsilon      - Small constant to add to denominator to avoid division
%                    by zero, in single precision.
%                    The default value is 2e-16.
%
%   The relative H1 loss is defined as:
%     loss = ||pred - gt||_{H^1} / ||gt||_{H^1}
%   where the H1 norm measures both function values and their gradients.
%   This was proposed by
%   Czarnecki, Wojciech M., et al. "Sobolev Training for Neural Networks."
%   Advances in Neural Information Processing Systems (2017).
%
%   Inputs PRED and GT must be dlarrays of identical size. They are
%   internally permuted to [S1, ..., SD, C, B] physical order before
%   computation.
%
%   The loss is calculated per sample in the batch and then reduced
%   according to the Reduction parameter.
%
%   Example:
%     B=2; C=1; S1=64; S2=64;
%     pred = dlarray(randn(S1,S2,C,B), 'SSCB');
%     gt = dlarray(randn(S1,S2,C,B), 'SSCB');
%     loss = relativeH1Loss(pred, gt);

% Copyright 2026 The MathWorks, Inc.

    arguments
        pred dlarray
        gt dlarray
        params.Normalize (1,1) logical = false
        params.SpatialSizes (1,:) double = []
        params.SquareRoot (1,1) logical = false
        params.Reduction (1,1) string {mustBeMember(params.Reduction, {'mean', 'sum', 'none'})} = "mean"
        params.Periodic (1,:) logical = true
        params.Epsilon (1, 1) single = 2e-16
    end

    if ~isequal(size(pred), size(gt))
        error('pred and gt must have identical size.');
    end

    pred = lossFunctions.permuteDimFirst(pred);
    gt = lossFunctions.permuteDimFirst(gt);

    sz = size(pred);
    nd = ndims(pred);
    D = nd - 2;

    if isempty(params.SpatialSizes)
        params.SpatialSizes = ones(1, D);
    elseif isscalar(params.SpatialSizes)
        params.SpatialSizes = repmat(params.SpatialSizes, 1, D);
    elseif numel(params.SpatialSizes) ~= D
        error('SpatialSizes must have length equal to the number of spatial dimensions.');
    end

    quadrature = params.SpatialSizes./sz(1:D);

    num = lossFunctions.h1Norm(gt - pred, ...
        Spacings=quadrature, ...
        Reduction='none', ...
        Normalize=params.Normalize, ...
        SquareRoot=params.SquareRoot, ...
        Periodic=params.Periodic);

    den = lossFunctions.h1Norm(gt, ...
        Spacings=quadrature, ...
        Reduction='none', ...
        Normalize=params.Normalize, ...
        SquareRoot=params.SquareRoot, ...
        Periodic=params.Periodic);

     loss = num./(den + params.Epsilon);

     switch params.Reduction
         case "mean"
             loss = mean(loss);
         case "sum"
             loss = sum(loss);
     end
end
