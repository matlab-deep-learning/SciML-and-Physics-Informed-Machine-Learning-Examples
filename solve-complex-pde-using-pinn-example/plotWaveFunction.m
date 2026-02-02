function plotWaveFunction(k,netU,crystalStructure)

nr = 100; % number of points per real space dimension
x = 2 * linspace(-.5,.5,nr);
y = 2 * linspace(-.5,.5,nr);

[X,Y] = meshgrid(x,y);
X = reshape(X,[],1);
Y = reshape(Y,[],1);
k = repmat(k,numel(X),1);
psi = forward(netU,[X,Y,k]);
psi = psi(:,1).^2+psi(:,2).^2;
psi = reshape(psi,nr,nr);

figure
contourf(x,y,psi)
axis equal
title("Wave Function ")
xlabel("x")
ylabel("y")

% scale of the color bar: from 0 (min) to twice the expected value for a
% constant wave function
areaUnitCell = cross([crystalStructure.a1 0],[crystalStructure.a2 0]);
areaUnitCell = vecnorm(areaUnitCell);
areaUnitCell = extractdata(areaUnitCell); % get double from dlarray
cmin = 0;
cmax = 2 / areaUnitCell;
clim([cmin,cmax])
colorbar

end