function [geomModule, domainIDs, boundaryIDs, volume, boundaryArea, ReferencePoint] = createBatteryModuleGeometry(numCellsInModule, cellWidth,cellThickness,tabThickness,tabWidth,cellHeight,tabHeight, connectorHeight )
%% Uses Boolean geometry functionality in PDE Toolbox, which requires release R2025a or later. 
% If you have an older version, use the helper function in this example: 
% https://www.mathworks.com/help/pde/ug/battery-module-cooling-analysis-and-reduced-order-thermal-model.html

% Copyright 2025 The MathWorks, Inc.

% First, create a single pouch cell by unioning the cell, tab and connector
% Cell creation
cell1 = fegeometry(multicuboid(cellThickness,cellWidth,cellHeight));
cell1 = translate(cell1,[cellThickness/2,cellWidth/2,0]);
% Tab creation
tab = fegeometry(multicuboid(tabThickness,tabWidth,tabHeight)); 
tabLeft = translate(tab,[cellThickness/2,tabWidth,cellHeight]);
tabRight = translate(tab,[cellThickness/2,cellWidth-tabWidth,cellHeight]);
% Union tabs to cells
geomPouch = union(cell1, tabLeft, KeepBoundaries=true);
geomPouch = union(geomPouch, tabRight, KeepBoundaries=true);
% Connector creation
overhang = (cellThickness-tabThickness)/2;
connector = fegeometry(multicuboid(tabThickness+overhang,tabWidth,connectorHeight));
connectorRight = translate(connector,[cellThickness/2+overhang/2,tabWidth,cellHeight+tabHeight]);
connectorLeft = translate(connector,[(cellThickness/2-overhang/2),cellWidth-tabWidth,cellHeight+tabHeight]);
% Union connectors to tabs
geomPouch = union(geomPouch,connectorLeft,KeepBoundaries=true); 
geomPouch = union(geomPouch,connectorRight,KeepBoundaries=true);
% Scale and translate completed pouch cell to create mirrored cell
geomPouchMirrored = translate(scale(geomPouch,[-1 1 1]),[cellThickness,0,0]);
% Union individual pouches to create full module
% Union even-numbered pouch cells together (original cells)
geomForward = fegeometry;
for i = 0:2:numCellsInModule-1
    offset = cellThickness*i;
    geom_to_append = translate(geomPouch,[offset,0,0]);
    geomForward = union(geomForward,geom_to_append);
end
% Union odd-numbered pouch cells together (mirrored cells)
geomBackward = fegeometry;
for i = 1:2:numCellsInModule-1
    offset = cellThickness*i;
    geom_to_append = translate(geomPouchMirrored,[offset,0,0]);
    geomBackward = union(geomBackward,geom_to_append);
end
% Union to create completed geometry module
geomModule = union(geomForward,geomBackward,KeepBoundaries=true);
% Rotate and translate the geometry
geomModule = translate(scale(geomModule,[1 -1 1]),[0 cellWidth 0]);
% Mesh the geometry to use query functions for identifying cells and faces
geomModule = generateMesh(geomModule,GeometricOrder="linear");
% Create Reference Points for each geometry future
ReferencePoint.Cell = [cellThickness/2,cellWidth/2,cellHeight/2];
ReferencePoint.TabLeft = [cellThickness/2,tabWidth,cellHeight+tabHeight/2];
ReferencePoint.TabRight = [cellThickness/2,cellWidth-tabWidth,cellHeight+tabHeight/2];
ReferencePoint.ConnectorLeft = [cellThickness/2,tabWidth,cellHeight+tabHeight+connectorHeight/2];
ReferencePoint.ConnectorRight = [cellThickness/2,cellWidth-tabWidth,cellHeight+tabHeight+connectorHeight/2];
% Helper function to get the cell IDs belonging to cell, tab and connector
[~,~,t] = meshToPet(geomModule.Mesh);
elementDomain = t(end,:);
tr = triangulation(geomModule.Mesh.Elements',geomModule.Mesh.Nodes');
getCellID = @(point,cellNumber) elementDomain(pointLocation(tr,point+(cellNumber(:)-1)*[cellThickness,0,0]));
% Helper function to get the volume of the cells, tabs, and connectors
getVolumeOneCell = @(geomCellID) geomModule.Mesh.volume(findElements(geomModule.Mesh,"region",Cell=geomCellID));
getVolume = @(geomCellIDs) arrayfun(@(n) getVolumeOneCell(n),geomCellIDs);
% Initialize cell ID and volume structs
domainIDs(1:numCellsInModule) = struct(Cell=[], ...
    TabLeft=[],TabRight=[], ...
    ConnectorLeft=[],ConnectorRight=[]); 
volume(1:numCellsInModule) = struct(Cell=[], ...
            TabLeft=[],TabRight=[], ...
            ConnectorLeft=[],ConnectorRight=[]);
% Helper function to get the IDs belonging to the left, right, front, back, top and bottom faces
getFaceID = @(offsetVal,offsetDirection,cellNumber) nearestFace(geomModule,...
    ReferencePoint.Cell + offsetVal/2 .*offsetDirection ... % offset ref. point to face
    + cellThickness*(cellNumber(:)-1)*[1,0,0]); % offset to cell
% Initialize face ID and area structs
boundaryIDs(1:numCellsInModule) = struct(FrontFace=[],BackFace=[], ...
    RightFace=[],LeftFace=[], ...
    TopFace=[],BottomFace=[]);
boundaryArea(1:numCellsInModule) = struct(FrontFace=[],BackFace=[], ...
    RightFace=[],LeftFace=[], ...
    TopFace=[],BottomFace=[]);
% Loop over cell, left tab, right tab, left connector, and right connector to get cell IDs and volumes
for part = string(fieldnames(domainIDs))'
    partid = num2cell(getCellID(ReferencePoint.(part),1:numCellsInModule));
    [domainIDs.(part)] = partid{:};
    volumesPart = num2cell(getVolume([partid{:}]));
    [volume.(part)] = volumesPart{:};
end
% Loop over front, back, right, left, top, and bottom faces IDs and areas
dimensions = [cellThickness;cellThickness;cellWidth;cellWidth;cellHeight;cellHeight];
vectors = [-1,0,0;1,0,0;0,1,0;0,-1,0;0,0,1;0,0,-1];
areaFormula = [cellHeight*cellWidth;cellHeight*cellWidth;cellThickness*cellHeight;cellThickness*cellHeight;cellThickness*cellWidth - tabThickness*tabWidth;cellThickness*cellWidth - tabThickness*tabWidth];
i = 1;
for face = string(fieldnames(boundaryIDs))'
    faceid = num2cell(getFaceID(dimensions(i),vectors(i,:),1:numCellsInModule));
    [boundaryIDs.(face)] = faceid{:};
    areasFace = num2cell(areaFormula(i)*ones(1,numCellsInModule));
    [boundaryArea.(face)] = areasFace{:};
    i = i+1;
end