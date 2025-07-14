function data = generatePendulumDataFNO(omega0,x0,numSamples,res,doPlot)
% Find the project root 
rootDir = findProjectRoot('startup.m'); 
dataDir = fullfile(rootDir, 'pendulumData');

% Create directory if it doesn't exist
if ~exist(dataDir, 'dir')
    mkdir(dataDir);
end
numModes = 5;
% generate forcing function data
for i = res
    [fSamples,thetaSamples,tGrid] = fnoDataHelper(numModes,numSamples,omega0,x0,i,doPlot);
    % save the data
    data.fSamples = fSamples;
    data.thetaSamples = thetaSamples;
    data.tGrid = tGrid;
    
    % Save the data
    save(fullfile(dataDir, sprintf('fno_data_%d.mat',i)), 'data');
    fprintf('FNO data of resolution %d written to fno_data_%d.mat \n',i,i);
end