% LicenseCC BY-NC-SA 4.0
% 
% Copyright (c) 2018 Andri Ashfahani Mahardhika Pratama

clc
clear
close all

%% load data
% dataset can be downloaded here:
% https://drive.google.com/open?id=1AG7u4-QSlNFAa9D4LiE7cvr-AXpu0L3z
% load rotatedmnist; I = 784;
% load permutedmnist; I = 784;
% load weather ; I = 8;
% load sea; I = 3;
% load hyperplane; I = 4;
% load rfid; I = 3;

%% This is ONLY for sample selection mode, in this paper we turned it off
selectiveSample.mode  = 0;      % 0: selective sample off, 1: selective sample on
selectiveSample.delta = 0.55;   % confidence level;

%% DEVDAN
portionOfLabeledData    = 1;      % 0-1 portion of labeled data
chunkSize               = 1000;   % number of data in a batch
mode                    = 0;      % 0: all components are on, 1: generative off, 
                                  % 2: growing hidden unit off, 3: pruning hidden unit off
[parameter,performance] = DEVDANplus(data,I,portionOfLabeledData,mode,...
    selectiveSample,chunkSize);
clear data
disp(performance)
