% MIT License
%
% Copyright (c) 2019 Andri Ashfahani Mahardhika Pratama
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

function [parameter,performance] = DEVDANplus(data,I,portionLabeledData,...
    mode,selectiveSample,chunkSize)
%% divide the data into nFolds chunks
fprintf('=========DEVDAN is started=========\n')
[nData,mn] = size(data);
M = mn - I;
l = 0;
nFolds = round(size(data,1)/chunkSize);                 % number of data chunk
chunk_size = round(nData/nFolds);
round_nFolds = floor(nData/chunk_size);
Data = {};
if round_nFolds == nFolds
    if nFolds == 1
        Data{1} = data;
    else
        for i=1:nFolds
            l=l+1;
            Data1 = data(((i-1)*chunk_size+1):i*chunk_size,:);
            Data{l} = Data1;
        end
    end
else
    if nFolds == 1
        Data{1} = data;
    else
        for i=1:nFolds-1
            l=l+1;
            Data1 = data(((i-1)*chunk_size+1):i*chunk_size,:);
            Data{l} = Data1;
        end
        foldplus = randperm(nFolds-1,1);
        Data{nFolds} = Data{foldplus};
    end
end
buffer_x = [];
buffer_T = [];
tTest = [];
clear data Data1

%% initiate model
K = 1;
if mode == 2
    K = M;
end
parameter.nn = netconfig([I K M]);
parameter.nn.M = M;
parameter.nn.index = 1;
parameter.nn.mode = mode;

%% initiate DAE parameter
parameter.dae{1}.lr = 0.001;
parameter.dae{1}.K = K;
parameter.dae{1}.Kg = 1;
parameter.dae{1}.kk = 0;
parameter.dae{1}.node = [];
parameter.dae{1}.BIAS2 = [];
parameter.dae{1}.VAR = [];
parameter.dae{1}.Loss = [];
parameter.dae{1}.miu_x_tail_old = 0;
parameter.dae{1}.var_x_tail_old = 0;
parameter.dae{1}.miu_NS_old = 0;
parameter.dae{1}.var_NS_old = 0;
parameter.dae{1}.miu_NHS_old = 0;
parameter.dae{1}.var_NHS_old = 0;
parameter.dae{1}.miumin_NS = [];
parameter.dae{1}.miumin_NHS = [];
parameter.dae{1}.stdmin_NS = [];
parameter.dae{1}.stdmin_NHS = [];
parameter.mode = 'sigmsigm';

%% initiate node evolving iterative parameters
parameter.ev{1}.kp = 0;
parameter.ev{1}.K = K;
parameter.ev{1}.Kd = 0;
parameter.ev{1}.node = [];
parameter.ev{1}.noded = [];
parameter.ev{1}.BIAS2 = [];
parameter.ev{1}.VAR = [];
parameter.ev{1}.miu_x_old = 0;
parameter.ev{1}.var_x_old = 0;
parameter.ev{1}.miu_NS_old = 0;
parameter.ev{1}.var_NS_old = 0;
parameter.ev{1}.miu_NHS_old = 0;
parameter.ev{1}.var_NHS_old = 0;
parameter.ev{1}.miumin_NS = [];
parameter.ev{1}.miumin_NHS = [];
parameter.ev{1}.stdmin_NS = [];
parameter.ev{1}.stdmin_NHS = [];

%% main loop, prequential evaluation
for t = 1:nFolds
    %% load the data chunk-by-chunk
    [bd,~] = size(Data{t}(:,I+1:mn));
    x = Data{t}(:,1:I);
    T = Data{t}(:,I+1:mn);
    tTarget(bd*t+(1-bd):bd*t,:) = T;
    clear Data{t}
    
    %% neural network testing
    start_test = tic;
    fprintf('=========Chunk %d of %d=========\n', t, size(Data,2))
    disp('Discriminative Testing: running ...');
    parameter.nn.t = t;
    parameter.nn = nettestparallel(parameter.nn,x,T,parameter.ev);
    
    %% metrics calculation
    parameter.ev{parameter.nn.index}.t = t;
    parameter.Loss(t) = parameter.nn.L(parameter.nn.index);
    tTest(bd*t+(1-bd):bd*t,:) = parameter.nn.as{1};
    act(bd*t+(1-bd):bd*t,:) = parameter.nn.act;
    out(bd*t+(1-bd):bd*t,:) = parameter.nn.out;
    parameter.residual_error(bd*t+(1-bd):bd*t,:) = parameter.nn.residual_error;
    parameter.cr(t) = parameter.nn.cr;
    ClassificationRate(t) = mean(parameter.cr);
    fprintf('Classification rate %d\n', ClassificationRate(t))
    disp('Discriminative Testing: ... finished');
    
    %% statistical measure
    [parameter.nn.f_measure(t,:),parameter.nn.g_mean,parameter.nn.recall(t,:),parameter.nn.precision(t,:),parameter.nn.err(t,:)] = stats(parameter.nn.act, parameter.nn.out, M);
    if t == nFolds - 1
        fprintf('=========DEVDAN is finished=========\n')
        break               % last chunk only testing
    end
    parameter.nn.test_time(t) = toc(start_test);
    
    if t > 1
        parameter.nn = WeightQuantization(parameter.nn,1);
    end
    
    %% Generative training
    start_train = tic;
    if mode ~= 1
        [parameter,~] = evdae_parallel(x,parameter);
        parameter.dae{1}.Loss(t) = parameter.dae{1}.LF;
        
        %% calculate bias^2/var based on generative training
        parameter.bs2g(t) = sum(parameter.dae{parameter.nn.index}.BIAS2(bd*t+(1-bd):bd*t,:))/bd;
        parameter.varg(t) = sum(parameter.dae{parameter.nn.index}.VAR  (bd*t+(1-bd):bd*t,:))/bd;
        
        %% calculate hidden node evolution based on generative training
        parameter.ndcg(t) = parameter.dae{parameter.nn.index}.K;
    end
    
    %% self labelling
    
    
    %% Discrinimanive training
    disp('Discriminative Training: running ...');
    parameter = nettrainparallel(parameter,T,portionLabeledData,...
        selectiveSample);
    disp('Discriminative Training: ... finished');
    parameter.nn.update_time(t) = toc(start_train);
    
    %% calculate bias^2/var based on discriminative training
%     parameter.bs2d(t) = sum(parameter.ev{parameter.nn.index}.BIAS2(bd*t+(1-bd):bd*t,:))/bd;
%     parameter.vard(t) = sum(parameter.ev{parameter.nn.index}.VAR  (bd*t+(1-bd):bd*t,:))/bd;
    
    %% calculate hidden node evolution based on discriminative training
    parameter.ndcd(t) = parameter.ev{parameter.nn.index}.K;
    
    %% clear current chunk data
    clear Data{t}
    parameter.nn.a = {};
end
clc

%% statistical measure
[performance.f_measure,performance.g_mean,performance.recall,performance.precision,performance.err] = stats(act, out, M);

parameter.nFolds = nFolds;
performance.update_time = [sum(parameter.nn.update_time) mean(parameter.nn.update_time) std(parameter.nn.update_time)];
performance.test_time = [sum(parameter.nn.test_time) mean(parameter.nn.test_time) std(parameter.nn.test_time)];
performance.classification_rate = [mean(parameter.cr(2:end)) std(parameter.cr(2:end))]*100;
performance.LayerWeight = parameter.nn.beta;
performance.NoOfnode = [mean(parameter.ev{1}.node) std(parameter.ev{1}.node)];
performance.NumberOfParameters = parameter.nn.mnop;
performance.compressionRate = parameter.ev{1}.kp/parameter.dae{1}.kk;

% plot(ClassificationRate)
% ylim([0 1.1]);
% xlim([1 nFolds]);
% ylabel('Classification Rate')
% xlabel('chunk');
% hold off
% figure
% plotconfusion(tTarget(2:end,:)',tTest(2:end,:)');
end

%% weight quantization
function nn = WeightQuantization(nn,nHiddenLayer)
for iHiddenLayer = nHiddenLayer
    [nn.W{iHiddenLayer},nn.trackW{iHiddenLayer}] = zeroedWeight(nn.W{iHiddenLayer},nn.trackW{iHiddenLayer});
    [nn.Ws{iHiddenLayer},nn.trackWs{iHiddenLayer}] = zeroedWeight(nn.Ws{iHiddenLayer},nn.trackWs{iHiddenLayer});
end
if nHiddenLayer == nn.hl
    %% prune node with 0 synapses
    nn = pruneNodeSyn(nn,nHiddenLayer);
end
end

%% prune node with no synapses
function net = pruneNodeSyn(net,wl)
checkWs = net.Ws{wl}(:,2:end);
delList = find(sum(checkWs)==0);
if numel(delList) > 0
    fprintf('%d nodes are PRUNED \n', numel(delList))
    net.W{wl}(delList,:) = [];
    net.dW{wl}(delList,:) = [];
    net.vW{wl}(delList,:) = [];
    net.trackW{wl}(delList,:) = [];
    net.Ws{wl}(:,delList+1)  = [];
    net.dWs{wl}(:,delList+1)  = [];
    net.vWs{wl}(:,delList+1)  = [];
    net.trackWs{wl}(:,delList+1) = [];
end
end

%% prune synapses
function [W,trackW] = zeroedWeight(W,trackW)
abs_W = abs(W);
mean_W = mean(mean(abs_W));
std_W = std(std(abs_W));
Threshold = mean_W - std_W;
if Threshold <= 0
    Threshold = mean_W - 0.5*std_W;
    if Threshold <= 0
        Threshold = mean_W - 0.25*std_W;
        if Threshold <= 0
            Threshold = mean_W - 0.1*std_W;
        end
    end
end
zeroedAbsWeight = abs_W;
zeroedAbsWeight(zeroedAbsWeight < Threshold) = 0;
zeroedAbsWeight(zeroedAbsWeight ~= 0) = 1;
W = W.*zeroedAbsWeight;
trackW = trackW.*zeroedAbsWeight;
end

function [inputData,targetData,nData] = reduceShuffleData(inputData,...
    targetData,nData,dataProportion,selectiveSample,classProbability)
nNewData = round(nData*dataProportion);
if selectiveSample.mode == 0
    s = RandStream('mt19937ar','Seed',0);
    ApplyPermutation = randperm(s,nData);
    inputData  = inputData(ApplyPermutation,:);
    targetData = targetData(ApplyPermutation,:);
    if dataProportion ~= 1
        noOfLabeledData = round(dataProportion*nData);
        inputData  = inputData(1:noOfLabeledData,:);
        targetData = targetData(1:noOfLabeledData,:);
    end
elseif selectiveSample.mode == 1
    nData = size(classProbability,1);
    selectedIndices = [];
    iIndices = 0;
    for iData = 1:nData
        if iIndices <= round(nData*dataProportion)
            confCandidate = sort(classProbability(iData,:),'descend');
            y1 = confCandidate(1);
            y2 = confCandidate(2);
            confFinal = y1/(y1+y2);
            if confFinal < selectiveSample.delta
                iIndices = iIndices + 1;
                selectedIndices(iIndices) = iData;
            end
        end
    end
    originalData   = inputData;
    originalTarget = targetData;
    inputData  = inputData(selectedIndices,:);
    targetData = targetData(selectedIndices,:);
    originalData(selectedIndices,:) = [];
    originalTarget(selectedIndices,:) = [];
end
[nData,~] = size(inputData);
if nData < nNewData
    nDataNeded = nNewData - nData;
    nCandidateData = size(originalData,1);
    indices = randperm(nCandidateData,nDataNeded);
    additionalData = originalData(indices,:);
    additionalTarget = originalTarget(indices,:);
    inputData = [inputData;additionalData];
    targetData = [targetData;additionalTarget];
    [nData,~] = size(inputData);
elseif nData > nNewData
    indices = randperm(nData,nNewData);
    inputData  = inputData(indices,:);
    targetData = targetData(indices,:);
    [nData,~] = size(inputData);
end
end

%% initialize network parameter
% This code aims to construct neural network with several hidden layer
% one can choose to either connect every hidden layer output to
% the last output or not

function nn = netconfig(layer)
nn.size                 = layer;
nn.n                    = numel(nn.size);  %  Number of layer
nn.hl                   = nn.n - 2;        %  number of hidden layer
nn.activation_function  = 'sigm';          %  Activation functions of hidden layers: 'sigm' (sigmoid) or 'tanh_opt' (optimal tanh).
nn.learningRate         = 0.01;             %  learning rate Note: typically needs to be lower when using 'sigm' activation function and non-normalized inputs.
nn.momentum             = 0.95;             %  Momentum
nn.outputConnect        = 1;               %  1: connect all hidden layer output to output layer, otherwise: only the last hidden layer is connected to output
nn.output               = 'softmax';       %  output unit 'sigm' (=logistic), 'softmax' and 'linear'

%% initiate weights and weight momentum for hidden layer
for i = 2 : nn.n - 1
    nn.W {i - 1} = normrnd(0,sqrt(2/(nn.size(i-1)+1)),[nn.size(i),nn.size(i - 1)+1]);
    nn.trackW {i - 1} = ones(size(nn.W {i - 1}));
    nn.vW{i - 1} = zeros(size(nn.W{i - 1}));
    nn.dW{i - 1} = zeros(size(nn.W{i - 1}));
    nn.c{i - 1} = normrnd(0,sqrt(2/(nn.size(i-1)+1)),[nn.size(i - 1),1]);
end

%% initiate weights and weight momentum for output layer
if nn.outputConnect == 1
    for i = 1 : nn.hl
        nn.Ws {i} = normrnd(0,sqrt(2/(size(nn.W{i},1)+1)),[nn.size(end),nn.size(i+1)+1]);
        nn.trackWs {i} = ones(size(nn.Ws {i}));
        nn.vWs{i} = zeros(size(nn.Ws{i}));
        nn.dWs{i} = zeros(size(nn.Ws{i}));
        nn.beta(i) = 1;
        nn.betaOld(i) = 1;
        nn.p(i) = 1;
    end
else
    nn.Ws  = normrnd(0,sqrt(2/(size(nn.W {i - 1},1)+1)),[nn.size(end),nn.size(end - 1)+1]);
    nn.trackWs {i - 1} = ones(size(nn.Ws {i}));
    nn.vWs = zeros(size(nn.Ws));
    nn.dWs = zeros(size(nn.Ws));
end
end

%% testing
function [nn] = nettestparallel(nn, x, T, ev)
%% feedforward
nn = netfeedforward(nn, x, T);
[m1,~] = size(T);

%% obtain trueclass label
[~,act] = max(T,[],2);

%% calculate the number of parameter
[a,b] = size(nn.W{1});
[c,d] = size(nn.Ws{1});
nop(1) = a*b + c*d;

%% calculate the number of node in each hidden layer
nn.nodes{1}(nn.t) = ev{1}.K;
nn.nop(nn.t) = sum(nop) + length(nn.c);
nn.mnop = [mean(nn.nop) std(nn.nop)];

%% calculate classification rate
[raw_out,out] = max(nn.as{1},[],2);
nn.bad = find(out ~= act);
nn.cr = 1 - numel(nn.bad)/m1;
nn.residual_error = 1 - raw_out;
nn.out = out;
nn.act = act;
end

%% evolving denoising autoencoder
function [parameter,h] = evdae_parallel(x,parameter)
ly = parameter.nn.index;
% [N,~] = size(x);

% %% add adversarial sample
% if parameter.nn.t > 2
%     sign_grad_Loss = sign(parameter.dae{ly}.Loss(parameter.nn.t-1) - parameter.dae{ly}.Loss(parameter.nn.t-2));
%     adversarial_samples = x + 0.007*sign_grad_Loss;
%     kk = randperm(N);
%     adversarial_samples = adversarial_samples(kk,:);
%     kk = randperm(N);
%     x = x(kk,:);
%     x = [x;adversarial_samples];
% end

%% initiate parameter
[N,~] = size(x);
% kk = randperm(N);
% x = x(kk,:);
[M,~] = size(parameter.nn.Ws{1});
[~,I] = size(x);
W = parameter.nn.W{ly}(:,2:end);
bb = size(W,2);
b = parameter.nn.W{ly}(:,1);
c = parameter.nn.c{ly};
[K,~] = size(W);
mode = parameter.mode;

%% a parameter to indicate if there is growing/pruning
grow = 0;
prune = 0;
addNode = parameter.nn.M;

%% initiate performance matrix
miu_x_tail_old = parameter.dae{ly}.miu_x_tail_old;
var_x_tail_old = parameter.dae{ly}.var_x_tail_old;
miu_NS_old = parameter.dae{ly}.miu_NS_old;
var_NS_old = parameter.dae{ly}.var_NS_old;
miu_NHS_old = parameter.dae{ly}.miu_NHS_old;
var_NHS_old = parameter.dae{ly}.var_NHS_old;
miumin_NS = parameter.dae{ly}.miumin_NS;
miumin_NHS = parameter.dae{ly}.miumin_NHS;
stdmin_NS = parameter.dae{ly}.stdmin_NS;
stdmin_NHS = parameter.dae{ly}.stdmin_NHS;
nodeg = parameter.dae{ly}.node;
Kg = parameter.dae{ly}.Kg;
lr = parameter.dae{ly}.lr;
kk = parameter.dae{ly}.kk;
node = parameter.ev{ly}.node;
BIAS2 = parameter.dae{ly}.BIAS2;
VAR = parameter.dae{ly}.VAR;

%% main loop devdann
x_tail = x;
kprune = 0;
kgrow = 0;
n_in = parameter.nn.size(1);

%% Generative training
disp('DEVDAN Training: running ...');
for k = 1:N
    kk = kk + 1;
    if parameter.nn.outputConnect ~= 1
        kp = kp + 1;
    end
    
    %% Input masking
    maskingLevel = 0.1;
    x_tail(k,:) = maskingnoise(x_tail(k,:),I,maskingLevel);
    
    %% feedforward #1
    a = W*x_tail(k,:)' + b;
    h = sigmf(a,[1,0]);
    a_hat = W'*h + c;
    switch mode
        case 'sigmsigm'
            x_hat = sigmf(a_hat,[1,0]);
        case 'sigmafn'
            x_hat = a_hat;
    end
    x_hat = x_hat';
    
    %% calculate error
    e(k,:) = x(k,:) - x_hat;
    L(k,:) = 0.5*norm(e(k,:))^2;
    
    if parameter.nn.index == parameter.nn.hl
        %% Incremental calculation of x_tail mean and variance
        [miu_x_tail,std_x_tail,var_x_tail] = meanstditer(miu_x_tail_old,var_x_tail_old,x_tail(k,:),kk);
        miu_x_tail_old = miu_x_tail;
        var_x_tail_old = var_x_tail;
        
        %% Expectation of z
        py = probit(miu_x_tail,std_x_tail);
        Ey = sigmf(W*py' + b,[1,0]);
        lr = 0.001;
        lr = lr.*(Ey)/max(Ey);
        switch mode
            case 'sigmsigm'
                Ez = sigmf(W'*Ey + c,[1,0]);
                Ez2 = sigmf(W'*Ey.^2 + c,[1,0]);
            case 'sigmafn'
                Ez = W'*Ey + c;
                Ez2 = W'*Ey.^2 + c;
        end
        
        %% Network mean calculation
        bias2 = (Ez - x(k,:)').^2;
        ns = bias2;
        NS = mean(ns);
        
        %% Incremental calculation of NS mean and variance
        [miu_NS,std_NS,var_NS] = meanstditer(miu_NS_old,var_NS_old,NS,kk);
        BIAS2(kk,:) = miu_NS;
        miu_NS_old = miu_NS;
        var_NS_old = var_NS;
        miustd_NS = miu_NS + std_NS;
%         netper1(k,:) = miustd_NS;
        if kk <= 1 || grow == 1
            miumin_NS = miu_NS;
            stdmin_NS = std_NS;
        else
            if miu_NS < miumin_NS
                miumin_NS = miu_NS;
            end
            if std_NS < stdmin_NS
                stdmin_NS = std_NS;
            end
        end
        switch mode
            case 'sigmsigm'
                miustdmin_NS = miumin_NS + (1.3*exp(-NS)+0.7)*stdmin_NS;
            case 'sigmafn'
                miustdmin_NS = miumin_NS + (1.3*exp(-NS)+0.7)*stdmin_NS;
        end
        
        %% growing hidden unit
        if miustd_NS >= miustdmin_NS && kk > 1 && parameter.nn.mode ~= 2
            grow = 1;
            K = K + addNode;
            Kg = Kg + addNode;
            fprintf('There are %d  new nodes FORMED around sample %d\n', addNode, k)
            kgrow = kgrow + 1;
            node(kk) = K;
            nodeg(kk) = Kg;
            b = [b;normrnd(0,sqrt(2/(n_in+addNode)),[addNode,1])];
            W = [W;normrnd(0,sqrt(2/(n_in+addNode)),[addNode,bb])];
            lr= [lr;0.001];
            parameter.nn.trackW{ly} = [parameter.nn.trackW{ly};ones(addNode,bb+1)];
            parameter.nn.trackWs{ly} = [parameter.nn.trackWs{ly} ones(parameter.nn.size(end),addNode)];
            parameter.nn.vW{ly} = [parameter.nn.vW{ly};zeros(addNode,I+1)];
            parameter.nn.dW{ly} = [parameter.nn.dW{ly};zeros(addNode,I+1)];
            parameter.nn.Ws{ly} = [parameter.nn.Ws{ly} normrnd(0,sqrt(2/(K+addNode)),[parameter.nn.size(end),addNode])];
            parameter.nn.vWs{ly} = [parameter.nn.vWs{ly} zeros(M,addNode)];
            parameter.nn.dWs{ly} = [parameter.nn.dWs{ly} zeros(M,addNode)];
        else
            grow = 0;
            node(kk) = K;
            nodeg(kk) = Kg;
        end
        
        %% Network variance calculation
        var = Ez2 - Ez.^2;
        NHS = mean(var);
        
        %% Incremental calculation of NHS mean and variance
        [miu_NHS,std_NHS,var_NHS] = meanstditer(miu_NHS_old,var_NHS_old,NHS,kk);
        VAR(kk,:) = miu_NHS;
        miu_NHS_old = miu_NHS;
        var_NHS_old = var_NHS;
        miustd_NHS = miu_NHS + std_NHS;
        netper2(k,:) = miustd_NHS;
        if kk <= I + 1 || prune == 1
            miumin_NHS = miu_NHS;
            stdmin_NHS = std_NHS;
        else
            if miu_NHS < miumin_NHS
                miumin_NHS = miu_NHS;
            end
            if std_NHS < stdmin_NHS
                stdmin_NHS = std_NHS;
            end
        end
        switch mode
            case 'sigmsigm'
                miustdmin_NHS = miumin_NHS + 2*(1.3*exp(-NHS)+0.7)*stdmin_NHS;
            case 'sigmafn'
                miustdmin_NHS = miumin_NHS + 2*(1.3*exp(-NHS)+0.7)*stdmin_NHS;
        end
        
        %% pruning hidden unit
        if grow == 0 && Kg > 1 && miustd_NHS >= miustdmin_NHS && kk > I + 1 && parameter.nn.mode ~= 3
            HS = Ey;
            [~,BB] = min(HS);
            fprintf('The node no %d is PRUNED around sample %d\n', BB, k)
            prune = 1;
            kprune = kprune + 1;
            K = K - 1;
            Kg = Kg - 1;
            node(kk) = K;
            nodeg(kk) = Kg;
            b(BB) = [];
            W(BB,:) = [];
            lr(BB)  = [];
            parameter.nn.trackW{ly}(BB,:) = [];
            parameter.nn.trackWs{ly}(:,BB+1) = [];
            parameter.nn.vW{ly}(BB,:) = [];
            parameter.nn.dW{ly}(BB,:) = [];
            parameter.nn.Ws{ly}(:,BB+1) = [];
            parameter.nn.vWs{ly}(:,BB+1) = [];
            parameter.nn.dWs{ly}(:,BB+1) = [];
        else
            node(kk) = K;
            nodeg(kk) = Kg;
            prune = 0;
        end
        
        %% feedforward #2 executed if there is a hidden node changing
        if grow == 1 || prune == 1
            a = W*x_tail(k,:)' + b;
            h = sigmf(a,[1,0]);
            a_hat = W'*h + c;
            switch mode
                case 'sigmsigm'
                    x_hat = sigmf(a_hat,[1,0]);
                case 'sigmafn'
                    x_hat = a_hat;
            end
            x_hat = x_hat';
            e(k,:) = x(k,:) - x_hat;
            L(k,:) = 0.5*norm(e(k,:))^2;
        end
    end
    
    %% Backpropaagation of DAE, tied weight
    lr = 0.001;
    W_old   = W;
    dedxhat = -e(k,:);
    switch mode
        case 'sigmsigm'
            del_j = x_hat.*(1 - x_hat);
        case 'sigmafn'
            del_j = ones(1,length(x_hat));
    end
    d3      = dedxhat.*del_j;
    d_act   = (h.*(1 - h))';
    d2      = (d3 * W') .* d_act;
    dW2     = (d3' * h');
    dW1     = (d2' * x_tail(k,:));
    dW      = dW1 + dW2';
    W       = W_old - lr*dW.*parameter.nn.trackW{ly}(:,2:end);
    del_W   = del_j.*W.*d_act';
    dedb    = dedxhat*del_W';
    b       = b - lr*dedb'.*parameter.nn.trackW{ly}(:,1);
    dejdcj  = dedxhat.*del_j;
    c       = c - (lr*dejdcj)';
    clear dejdcj dedb del_W dW dW1 dW2 d2 d_act d3 del_j
end
a = W*x' + b;
yh = sigmf(a,[1,0]);
h = yh';

%% substitute the weight back to evdae
parameter.nn.W{ly} = [b W];
parameter.nn.c{ly} = c;
parameter.nn.K(ly) = K;
parameter.dae{ly}.node = nodeg;
parameter.ev{ly}.node = node;
parameter.dae{ly}.LF = mean(L);
parameter.dae{ly}.BIAS2 = BIAS2;
parameter.dae{ly}.VAR = VAR;
parameter.dae{ly}.kk = kk;
parameter.dae{ly}.lr = lr;
parameter.dae{ly}.K = K;
parameter.dae{ly}.Kg = Kg;
parameter.ev{ly}.K = K;
parameter.dae{ly}.miu_x_tail_old = miu_x_tail_old;
parameter.dae{ly}.var_x_tail_old = var_x_tail_old;
parameter.dae{ly}.miu_NS_old = miu_NS_old;
parameter.dae{ly}.var_NS_old = var_NS_old;
parameter.dae{ly}.miu_NHS_old = miu_NHS_old;
parameter.dae{ly}.var_NHS_old = var_NHS_old;
parameter.dae{ly}.miumin_NS = miumin_NS;
parameter.dae{ly}.miumin_NHS = miumin_NHS;
parameter.dae{ly}.stdmin_NS = stdmin_NS;
parameter.dae{ly}.stdmin_NHS = stdmin_NHS;

%% substitute the weight back to NN
disp('DEVDAN Training: ... finished');
end

function in = maskingnoise(in,nin,noiseIntensity)
%% input masking
if nin > 1
    if noiseIntensity > 0
        nMask = max(round(noiseIntensity*nin),1);
        mask_gen = randperm(nin,nMask);
        in(1,mask_gen) = 0;
    end
else
    mask_gen = rand(size(in(1,:))) > 0.3;
    in = in*mask_gen;
end
end

%% discriminative training
function parameter  = nettrainparallel(parameter,y,portionLabeledData,selectiveSample)
[~,bb] = size(parameter.nn.W{parameter.nn.index});
grow = 0;
prune = 0;
addNode = parameter.nn.M;
ly = parameter.nn.index;
x = parameter.nn.a{ly};
% [N,~] = size(x);

%% add adverarial samples
% if parameter.nn.t > 2
%     sign_grad_Loss = sign(parameter.Loss(parameter.nn.t-1) - parameter.Loss(parameter.nn.t-2));
%     adversarial_samples = x + 0.007*sign_grad_Loss;
%     kk = randperm(N);
%     x = x(kk,:);
%     y1 = y(kk,:);
%     
%     kk = randperm(N);
%     adversarial_samples = adversarial_samples(kk,:);
%     y2 = y(kk,:);
%     x = [x;adversarial_samples];
%     y = [y1;y2];
% end

%% initiate performance matrix
BIAS2 = parameter.ev{ly}.BIAS2;
VAR = parameter.ev{ly}.VAR;
miu_x_old = parameter.ev{ly}.miu_x_old;
var_x_old = parameter.ev{ly}.var_x_old;
miu_NS_old = parameter.ev{ly}.miu_NS_old;
var_NS_old = parameter.ev{ly}.var_NS_old;
miu_NHS_old = parameter.ev{ly}.miu_NHS_old;
var_NHS_old = parameter.ev{ly}.var_NHS_old;
miumin_NS = parameter.ev{ly}.miumin_NS;
miumin_NHS = parameter.ev{ly}.miumin_NHS;
stdmin_NS = parameter.ev{ly}.stdmin_NS;
stdmin_NHS = parameter.ev{ly}.stdmin_NHS;
t = parameter.ev{ly}.t;
kp = parameter.ev{ly}.kp;
K = parameter.ev{ly}.K;
Kd = parameter.ev{ly}.Kd;
node = parameter.ev{ly}.node;
noded = parameter.ev{ly}.noded;

%% initiate training model
net = netconfigtrain([1 1 1]);
net.activation_function = parameter.nn.activation_function;

%% substitute the weight to be trained to training model
net.W{1} = parameter.nn.W{ly};
net.trackW{1} = parameter.nn.trackW{ly};
net.vW{1} = parameter.nn.vW{ly};
net.dW{1} = parameter.nn.dW{ly};
net.W{2} = parameter.nn.Ws{ly};
net.trackW{2} = parameter.nn.trackWs{ly};
net.vW{2} = parameter.nn.vWs{ly};
net.dW{2} = parameter.nn.dWs{ly};

%% load the data for training
[N,I] = size(x);
classProbability = parameter.nn.as{1};
[x,y,N] = reduceShuffleData(x,y,N,portionLabeledData,selectiveSample,...
    classProbability);

%% xavier initialization
n_in = parameter.nn.size(1);

%% main loop, train the model
for k = 1 : N
    kp = kp + 1;
    
    %% Incremental calculation of x_tail mean and variance
    [miu_x,std_x,var_x] = meanstditer(miu_x_old,var_x_old,x(k,:),kp);
    miu_x_old = miu_x;
    var_x_old = var_x;
    
    %% Expectation of z
    py = probit(miu_x,std_x);
    Ey = sigmf(net.W{1}*py',[1,0]);
    Ey = [1;Ey];
    Ez = net.W{2}*Ey;
    %     Ez = exp(Ez-max(Ez,[],1));
    Ez = exp(Ez);
    Ez = Ez./sum(Ez);
    Ez2 = net.W{2}*Ey.^2;
    %     Ez2 = exp(Ez2-max(Ez2,[],1));
    Ez2 = exp(Ez2);
    Ez2 = Ez2./sum(Ez2);
    
    %% Network mean calculation
    bias2 = (Ez - y(k,:)').^2;
    ns = bias2;
    NS = norm(ns,'fro');
    
    %% Incremental calculation of NS mean and variance
    [miu_NS,std_NS,var_NS] = meanstditer(miu_NS_old,var_NS_old,NS,kp);
    miu_NS_old = miu_NS;
    var_NS_old = var_NS;
    miustd_NS = miu_NS + std_NS;
    miuNS(k,:) = miu_NS;
    if kp <= 1 || grow == 1
        miumin_NS = miu_NS;
        stdmin_NS = std_NS;
    else
        if miu_NS < miumin_NS
            miumin_NS = miu_NS;
        end
        if std_NS < stdmin_NS
            stdmin_NS = std_NS;
        end
    end
    miuminNS(k,:) = miumin_NS;
    miustdmin_NS = miumin_NS + (1.3*exp(-NS)+0.7)*stdmin_NS;
    BIAS2(kp,:) = miu_NS;
    
    %% growing hidden unit
    if miustd_NS >= miustdmin_NS && kp > 1 && parameter.nn.mode ~= 2
        grow = 1;
        K = K + addNode;
        Kd = Kd + addNode;
        fprintf('There are %d new nodes FORMED around sample %d\n', addNode, k)
        node(kp) = K;
        noded(kp) = Kd;
        net.W{1}  = [net.W{1};normrnd(0,sqrt(2/(n_in+1)),[addNode,bb])];
        net.trackW{1} = [net.trackW{1};ones(addNode,bb)];
        net.vW{1} = [net.vW{1};zeros(addNode,bb)];
        net.dW{1} = [net.dW{1};zeros(addNode,bb)];
        net.W{2}  = [net.W{2} normrnd(0,sqrt(2/(K+addNode)),[parameter.nn.size(end),addNode])];
        net.trackW{2} = [net.trackW{2} ones(parameter.nn.size(end),addNode)];
        net.vW{2} = [net.vW{2} zeros(parameter.nn.size(end),addNode)];
        net.dW{2} = [net.dW{2} zeros(parameter.nn.size(end),addNode)];
    else
        grow = 0;
        node(kp) = K;
        noded(kp) = Kd;
    end
    
    %% Network variance calculation
    var = Ez2 - Ez.^2;
    NHS = norm(var,'fro');
    
    %% Incremental calculation of NHS mean and variance
    [miu_NHS,std_NHS,var_NHS] = meanstditer(miu_NHS_old,var_NHS_old,NHS,kp);
    miu_NHS_old = miu_NHS;
    var_NHS_old = var_NHS;
    miustd_NHS = miu_NHS + std_NHS;
    miuNHS(k,:) = miu_NHS;
    if kp <= I+1 || prune == 1
        miumin_NHS = miu_NHS;
        stdmin_NHS = std_NHS;
    else
        if miu_NHS < miumin_NHS
            miumin_NHS = miu_NHS;
        end
        if std_NHS < stdmin_NHS
            stdmin_NHS = std_NHS;
        end
    end
    miuminNHS(k,:) = miumin_NHS;
    miustdmin_NHS = miumin_NHS + 2*(1.3*exp(-NHS)+0.7)*stdmin_NHS;
    VAR(kp,:) = miu_NHS;
    
    %% pruning hidden unit
    if grow == 0 && Kd > 0 && miustd_NHS >= miustdmin_NHS && kp > I + 1 && parameter.nn.mode ~= 3
        HS = Ey(2:end);
        [~,BB] = min(HS);
        fprintf('The node no %d is PRUNED around sample %d\n', BB, k)
        prune = 1;
        K = K - 1;
        Kd = Kd - 1;
        node(kp) = K;
        noded(kp) = Kd;
        net.W{1}(BB,:) = [];
        net.trackW{1}(BB,:) = [];
        net.vW{1}(BB,:) = [];
        net.dW{1}(BB,:) = [];
        net.W{2}(:,BB+1) = [];
        net.trackW{2}(:,BB+1) = [];
        net.vW{2}(:,BB+1) = [];
        net.dW{2}(:,BB+1) = [];
    else
        node(kp) = K;
        noded(kp) = Kd;
        prune = 0;
    end
    
    %% feedforward
    net = netffsingle(net, x(k,:), y(k,:));
    
    %% update
    net = netbackpropagation(net);
    net = netupdate(net);
end

%% substitute the weight back to main model
parameter.nn.W{ly}   = net.W{1};
parameter.nn.trackW{ly} = net.trackW{1};
parameter.nn.Ws{ly}  = net.W{2};
parameter.nn.trackWs{ly} = net.trackW{2};

%% reset momentum and gradient
parameter.nn.vW{ly}  = net.vW{1}*0;
parameter.nn.dW{ly}  = net.dW{1}*0;
parameter.nn.vWs{ly} = net.vW{2}*0;
parameter.nn.dWs{ly} = net.dW{2}*0;

%% substitute the recursive calculation
parameter.ev{ly}.t = t + 1;
parameter.ev{ly}.kp = kp;
parameter.ev{ly}.K = K;
parameter.ev{ly}.Kd = Kd;
% parameter.dae{ly}.K = K;
parameter.ev{ly}.node = node;
parameter.ev{ly}.noded = noded;
parameter.ev{ly}.BIAS2 = BIAS2;
parameter.ev{ly}.VAR = VAR;
parameter.ev{ly}.miu_x_old = miu_x_old;
parameter.ev{ly}.var_x_old = var_x_old;
parameter.ev{ly}.miu_NS_old = miu_NS_old;
parameter.ev{ly}.var_NS_old = var_NS_old;
parameter.ev{ly}.miu_NHS_old = miu_NHS_old;
parameter.ev{ly}.var_NHS_old = var_NHS_old;
parameter.ev{ly}.miumin_NS = miumin_NS;
parameter.ev{ly}.miumin_NHS = miumin_NHS;
parameter.ev{ly}.stdmin_NS = stdmin_NS;
parameter.ev{ly}.stdmin_NHS = stdmin_NHS;
end

%% feedforward
function nn = netfeedforward(nn, x, y)
n = nn.n;
m = size(x,1);
x = [ones(m,1) x];      % by adding 1 to the first coulomn, it means the first coulomn of W is bias
nn.a{1} = x;            % the first activity is the input itself

%% feedforward from input layer through all the hidden layer
for i = 2 : n-1
    switch nn.activation_function
        case 'sigm'
            nn.a{i} = sigmf(nn.a{i - 1} * nn.W{i - 1}',[1,0]);
        case 'relu'
            nn.a{i} = max(nn.a{i - 1} * nn.W{i - 1}',0);
    end
    nn.a{i} = [ones(m,1) nn.a{i}];
end

%% propagate to the output layer
for i = 1 : nn.hl
    switch nn.output
        case 'sigm'
            nn.as{i} = sigmf(nn.a{i + 1} * nn.Ws{i}',[1,0]);
        case 'linear'
            nn.as{i} = nn.a{i + 1} * nn.Ws{i}';
        case 'softmax'
            nn.as{i} = nn.a{i + 1} * nn.Ws{i}';
            nn.as{i} = exp( nn.as{i} - max(nn.as{i},[],2));
            nn.as{i} = nn.as{i}./sum(nn.as{i}, 2);
    end
    
    %% calculate error
    nn.e{i} = y - nn.as{i};
    
    %% calculate loss function
    switch nn.output
        case {'sigm', 'linear'}
            nn.L(i) = 1/2 * sum(sum(nn.e .^ 2)) / m;
        case 'softmax'
            nn.L(i) = -sum(sum(y .* log(nn.as{i}))) / m;
    end
end
end

%% probit function
function p = probit(miu,std)
p = (miu./(1 + pi.*(std.^2)./8).^0.5);
end

%% recursive mean and standard deviation
function [miu,std,var] = meanstditer(miu_old,var_old,x,k)
miu = miu_old + (x - miu_old)./k;
var = var_old + (x - miu_old).*(x - miu);
std = sqrt(var/k);
end

%% initialize network for training
function nn = netconfigtrain(layer)
nn.size   = layer;
nn.n      = numel(nn.size);
nn.activation_function              = 'sigm';       %  Activation functions of hidden layers: 'sigm' (sigmoid) or 'tanh_opt' (optimal tanh).
nn.learningRate                     = 0.01;  %2      %  learning rate Note: typically needs to be lower when using 'sigm' activation function and non-normalized inputs.
nn.momentum                         = 0.95;          %  Momentum
nn.output                           = 'softmax';    %  output unit 'sigm' (=logistic), 'softmax' and 'linear'
end

%% feedforward of a single hidden layer network
function nn = netffsingle(nn, x, y)

n = nn.n;
m = size(x,1);
nn.a{1} = x;

%% feedforward from input layer through all the hidden layer
for i = 2 : n-1
    switch nn.activation_function
        case 'sigm'
            nn.a{i} = sigmf(nn.a{i - 1} * nn.W{i - 1}',[1,0]);
        case 'relu'
            nn.a{i} = max(nn.a{i - 1} * nn.W{i - 1}',0);
    end
    nn.a{i} = [ones(m,1) nn.a{i}];
end

%% propagate to the output layer
switch nn.output
    case 'sigm'
        nn.a{n} = sigmf(nn.a{n - 1} * nn.W{n - 1}',[1,0]);
    case 'linear'
        nn.a{n} = nn.a{n - 1} * nn.W{n - 1}';
    case 'softmax'
        nn.a{n} = nn.a{n - 1} * nn.W{n - 1}';
        nn.a{n} = exp( nn.a{n} - max(nn.a{n},[],2));
        nn.a{n} = nn.a{n}./sum(nn.a{n}, 2);
end

%% calculate error
nn.e = y - nn.a{n};
end

%% backpropagation
function nn = netbackpropagation(nn)
n = nn.n;
switch nn.output
    case 'sigm'
        d{n} = - nn.e .* (nn.a{n} .* (1 - nn.a{n}));
    case {'softmax','linear'}
        d{n} = - nn.e;          % dL/dy
end

for i = (n - 1) : -1 : 2
    switch nn.activation_function
        case 'sigm'
            d_act = nn.a{i} .* (1 - nn.a{i}); % contains b
        case 'tanh_opt'
            d_act = 1.7159 * 2/3 * (1 - 1/(1.7159)^2 * nn.a{i}.^2);
        case 'relu'
            d_act = zeros(1,length(nn.a{i}));
            d_act(nn.a{i}>0) = 1;
    end
    
    if i+1 == n
        d{i} = (d{i + 1} * nn.W{i}) .* d_act;
    else
        d{i} = (d{i + 1}(:,2:end) * nn.W{i}) .* d_act;
    end
end

for i = 1 : (n - 1)
    if i + 1 == n
        nn.dW{i} = (d{i + 1}' * nn.a{i});
    else
        nn.dW{i} = (d{i + 1}(:,2:end)' * nn.a{i});
    end
end
end

%% weight update
function nn = netupdate(nn)
for i = 1 : (nn.n - 1)
    dW = nn.dW{i}.*nn.trackW{i};%+2*0.000125*nn.W{i};
    dW = nn.learningRate * dW;
    if(nn.momentum > 0)
        nn.vW{i} = nn.momentum*nn.vW{i} + dW;
        dW = nn.vW{i};
    end
    nn.W{i} = nn.W{i} - dW;
    
    
end
%% reset momentum and gradient
% nn.dW{i} = nn.dW{i}*0;
% nn.vW{i} = nn.vW{i}*0;
end

%% statistical measure
function [f_measure,g_mean,recall,precision,err] = stats(f, h, mclass)
%   [f_measure,g_mean,recall,precision,err] = stats(f, h, mclass)
%     @f - vector of true labels
%     @h - vector of predictions on f
%     @mclass - number of classes
%     @f_measure
%     @g_mean
%     @recall
%     @precision
%     @err
%

%     stats.m
%     Copyright (C) 2013 Gregory Ditzler
% 
%     This program is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.
% 
%     This program is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
% 
%     You should have received a copy of the GNU General Public License
%     along with this program.  If not, see <http://www.gnu.org/licenses/>.

F = index2vector(f, mclass);
H = index2vector(h, mclass);

recall = compute_recall(F, H, mclass);
err = 1 - sum(diag(H'*F))/sum(sum(H'*F));
precision = compute_precision(F, H, mclass);
g_mean = compute_g_mean(recall, mclass);
f_measure = compute_f_measure(F, H, mclass);


function g_mean = compute_g_mean(recall, mclass)
g_mean = (prod(recall))^(1/mclass);
end

function f_measure = compute_f_measure(F, H, mclass)
f_measure = zeros(1, mclass);
for c = 1:mclass
  f_measure(c) = 2*F(:, c)'*H(:, c)/(sum(H(:, c)) + sum(F(:, c))); 
end
f_measure(isnan(f_measure)) = 1;
end

function precision = compute_precision(F, H, mclass)
precision = zeros(1, mclass);
for c = 1:mclass
  precision(c) = F(:, c)'*H(:, c)/sum(H(:, c)); 
end
precision(isnan(precision)) = 1;
end

function recall = compute_recall(F, H, mclass)
recall = zeros(1, mclass);
for c = 1:mclass
  recall(c) = F(:, c)'*H(:, c)/sum(F(:, c)); 
end
recall(isnan(recall)) = 1;
end

function y = index2vector(x, mclass)
y = zeros(numel(x), mclass);
for n = 1:numel(x)
  y(n, x(n)) = 1;
end
end
end