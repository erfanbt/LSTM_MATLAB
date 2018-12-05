%% 
clear;
load('story_1.mat');
% data = chickenpox_dataset;
% data = [data{:}];
% 
% figure
% plot(data)
% xlabel("Month")
% ylabel("Cases")
% title("Monthy Cases of Chickenpox")
%% 
% numTimeStepsTrain = floor(0.9*112);
% XTrain = V(1:numTimeStepsTrain,:)';
% YTrain = V(2:numTimeStepsTrain+1,:)';
% XTest = V(numTimeStepsTrain+1:end-1,:)';
% YTest = V(numTimeStepsTrain+2:end,:)';
X = V(1:111,:)';
Y = V(2:112,:)';
%% 
% mu = mean(XTrain);
% sig = std(XTrain);
% 
% XTrain = (XTrain - mu) / sig;
% YTrain = (YTrain - mu) / sig;
% 
% XTest = (XTest - mu) / sig;
%% 
inputSize = 50;
numResponses = 50;
numHiddenUnits = 200;

layers = [ ...
    sequenceInputLayer(inputSize)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];
%% 
opts = trainingOptions('adam', ...
    'MaxEpochs',1000, ...
    'GradientThreshold',0.01, ...
    'InitialLearnRate',0.01, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',500, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');
%% 
% net = trainNetwork(XTrain,YTrain,layers,opts); 
net = trainNetwork(X,Y,layers,opts);
%% 
% net = predictAndUpdateState(net,XTrain);
% [net,YPred(:,1)] = predictAndUpdateState(net,YTrain(:,end));

% numTimeStepsTest = numel(XTest);
% for i = 2:11
%     [net,YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-1));
% end
[net,YPred] = predictAndUpdateState(net,X);
%% 
% YPred = sig*YPred + mu;
% YTest = (YTest - mu) / sig;
%% 
for i=1:111
n_YPred(i)=norm(YPred(:,i));
n_Y(i)=norm(Y(:,i));
end
rmse = norm(YPred-Y);
%% 
% figure
% plot(data(1:numTimeStepsTrain))
% hold on
% idx = numTimeStepsTrain:(numTimeStepsTrain+numTimeStepsTest);
% plot(idx,[data(numTimeStepsTrain) YPred],'.-')
% hold off
% xlabel("Month")
% ylabel("Cases")
% title("Forecast")
% legend(["Observed" "Forecast"])
%% 
figure
subplot(2,1,1)
plot(n_Y)
hold on
plot(n_YPred,'.-')
hold off
legend(["Observed" "Predicted"])
ylabel("Cases")
title("Forecast with Updates")

subplot(2,1,2)
stem(n_YPred - n_Y)
xlabel("Month")
ylabel("Error")
title("RMSE = " + rmse)