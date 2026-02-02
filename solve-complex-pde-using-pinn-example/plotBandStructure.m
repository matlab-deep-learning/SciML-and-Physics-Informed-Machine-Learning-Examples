function plotBandStructure(kPts,E,bzBoundaryPosition)
eGroundTruth = freeEnergy(kPts);
plot(bzBoundaryPosition,E(1:length(bzBoundaryPosition),:),'.')
hold on
plot(bzBoundaryPosition,eGroundTruth(1:length(bzBoundaryPosition),:),'.')
hold off
legend(["Neural Network" "Free Electrons"])
xticks([0 1 2 2+sqrt(2)])
xticklabels(["G" "X" "M" "G"])
grid("on")
end

function E = freeEnergy(k)
E = 0.5 * (k(:,1).^2+k(:,2).^2);
end