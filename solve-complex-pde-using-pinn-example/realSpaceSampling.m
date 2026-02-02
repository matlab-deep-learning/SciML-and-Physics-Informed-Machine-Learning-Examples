function rPts = realSpaceSampling(crystalStructure)
rPts = (rand(crystalStructure.nPoints,1)-0.5) .* crystalStructure.a1 + (rand(crystalStructure.nPoints,1)-0.5) .* crystalStructure.a2;
end