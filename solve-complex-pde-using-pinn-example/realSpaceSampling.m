function rPts = realSpaceSampling(crystalStructure)
% Copyright 2026 The MathWorks, Inc.
rPts = (rand(crystalStructure.nPoints,1)-0.5) .* crystalStructure.a1 + (rand(crystalStructure.nPoints,1)-0.5) .* crystalStructure.a2;

end
