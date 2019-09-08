% MIT License
% 
% Copyright (c) 2018 Andri Ashfahani Mahardhika Pratama
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
% SOFTWARE.

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
chunkSize = 1000;       % no of data in a batch
epoch = 1;              % no of epoch
[parameter,performance] = ADLplus(data,I,chunkSize,epoch);
clear data
disp(performance)

% The classification rate in each chunk can be seen in parameter.cr
% The results are the average of all timestamps