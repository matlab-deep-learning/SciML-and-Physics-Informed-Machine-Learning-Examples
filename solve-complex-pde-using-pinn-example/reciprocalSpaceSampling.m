function [kPts,bzBoundaryPosition] = reciprocalSpaceSampling(crystalStructure)
% 1. Randomly sample along high-symmetry boundary of IBZ.
% 2. Randomly sample inside IBZ.
% 3. Randomly shift either into adjacent higher-order Brillouin zones by adding integer multiples of reciprocal lattice vectors - this is how to get higher-energy states.

% High symmetry points
Gamma = [0 0]; % center of the BZ
X = crystalStructure.b1/2; % center of the BZ boundary
M = (crystalStructure.b1+crystalStructure.b2)/2; % corner of the BZ boundary

% Number of points
nGX = round(crystalStructure.nPoints/10);     % number of points between gamma and X
nXM = round(crystalStructure.nPoints/10);     % number of points between X and M
nMG = round(crystalStructure.nPoints/10);     % number of points between M and G
nIBZ = crystalStructure.nPoints - nGX - nXM - nMG;    % number of points inside the IBZ

%% Sample along boundary.
% Gamma -> X
randGX = rand(nGX,1);
ptsGX = Gamma + randGX * (X-Gamma);
% X -> M
randXM = rand(nXM,1);
ptsXM = X + randXM * (M-X);
% M -> Gamma
randMG = rand(nMG,1);
ptsMG = M + randMG * (Gamma-M);

%% Sample inside IBZ.
% Sample inside a triangle by sampling inside a diamond and
% then reflecting one half onto the other along the diagonal.
randomDiamond = (rand(nIBZ,1)*M + rand(nIBZ,1)*X);
inTriangle = randomDiamond(:,1) < pi/crystalStructure.a; % specific to square
randomDiamond(~inTriangle,:) = M + X - randomDiamond(~inTriangle,:); % reflect back into triangle
ptsIBZ = randomDiamond;

%% Concatenate everything.
kPts = [ptsGX; ptsXM; ptsMG; ptsIBZ];

%% Shift some points randomly into surrounding unit cells.
N = nGX + nXM + nMG + nIBZ;
kPts = kPts + randi([-1,1],N,1)*crystalStructure.b1 + randi([-1,1],N,1)*crystalStructure.b2;

%% Save the boundary position of edge points - this is used to plot the band structure.
bzBoundaryPosition = [randGX;1+randXM;2+sqrt(2)*randMG];
end