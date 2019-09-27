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

%% run stacked autonomous deep learning
portionOfLabeledData    = 1;     % 0-1 portion of labeled data
chunkSize               = 1000;  % no of data in a batch
epoch                   = 1;     % no of epoch
[parameter,performance] = ADLplus(data,I,chunkSize,epoch,...
    portionOfLabeledData);
clear data
disp(performance)

% The classification rate in each chunk can be seen in parameter.cr
% The results are the average of all timestamps
