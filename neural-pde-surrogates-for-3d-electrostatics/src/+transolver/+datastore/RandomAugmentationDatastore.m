classdef RandomAugmentationDatastore < matlab.io.Datastore & ...
          matlab.io.datastore.Subsettable
  % RandomAugmentationDatastore Datastore that applies random rotations and reflections.
  %
  %   ds = transolver.datastore.RandomAugmentationDatastore(inputFeatures)
  %   creates a datastore that applies random augmentations to
  %   3-D input features on each read. Augmentations include random
  %   rotation around the Z-axis and random reflection across X, which are
  %   valid for axisymmetric geometries.
  %
  %   Input:
  %       inputFeatures - 3-D array of size C-by-N-by-NumSamples, where
  %                       the first three channels are spatial coordinates.

  %   Copyright 2026 The MathWorks, Inc.

  properties
      InputFeatures
      Index
  end
  methods
      function this = RandomAugmentationDatastore(inputFeatures)
          this.InputFeatures = inputFeatures;
          this.Index = 1;
      end
      function tf = hasdata(this)
          tf = this.Index<=size(this.InputFeatures,3);
      end
      function [data,info] = read(this)
          data = this.InputFeatures(:,:,this.Index);
          % Random rotation around Z-axis (valid for axisymmetric geometry)
          theta = 360*rand();
          ct = cosd(theta); st = sind(theta);
          R = [ct -st 0; st ct 0; 0 0 1];
          data(1:3,:,:) = R * data(1:3,:,:);
          % Random reflection across x (valid for axisymmetric geometry)
          if rand() > 0.5
              data(1,:,:) = -data(1,:,:);
          end
          this.Index = this.Index + 1;
          info = [];
      end
      function reset(this)
          this.Index = 1;
      end
  end
  methods(Access=protected)
      function n = maxpartitions(this)
          n = size(this.InputFeatures,3);
      end
      function subds = subsetByReadIndices(this,idx)
          subds = copy(this);
          subds.InputFeatures = subds.InputFeatures(:,:,idx);
      end
  end
end