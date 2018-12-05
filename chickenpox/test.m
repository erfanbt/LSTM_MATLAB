%%
clear;
load('network.mat');
%% 
% netn.Layers(1, 1).Name = net.Layers(1, 1).Name;
% netn.Layers(1, 1).InputSize = net.Layers(1, 1).InputSize;
% netn.Layers(2, 1).Name = net.Layers(2, 1).Name;
% netn.Layers(2, 1).InputSize = net.Layers(2, 1).InputSize;
% netn.Layers(2, 1).NumHiddenUnits = net.Layers(2, 1).NumHiddenUnits;
% netn.Layers(2, 1).OutputMode = net.Layers(2, 1).OutputMode;
netn.Layers(2, 1).InputWeights = net.Layers(2, 1).InputWeights;
% netn.Layers(2, 1).InputWeightsLearnRateFactor = net.Layers(2, 1).InputWeightsLearnRateFactor;
% netn.Layers(2, 1).InputWeightsL2Factor = net.Layers(2, 1).InputWeightsL2Factor;
netn.Layers(2, 1).RecurrentWeights = net.Layers(2, 1).RecurrentWeights;
% netn.Layers(2, 1).RecurrentWeightsLearnRateFactor = net.Layers(2, 1).RecurrentWeightsLearnRateFactor;
% netn.Layers(2, 1).RecurrentWeightsL2Factor = net.Layers(2, 1).RecurrentWeightsL2Factor;
netn.Layers(2, 1).Bias = net.Layers(2, 1).Bias;
% netn.Layers(2, 1).BiasLearnRateFactor = net.Layers(2, 1).BiasLearnRateFactor;
% netn.Layers(2, 1).BiasL2Factor = net.Layers(2, 1).BiasL2Factor;
% netn.Layers(2, 1).HiddenState = net.Layers(2, 1).HiddenState;
% netn.Layers(2, 1).CellState = net.Layers(2, 1).CellState;
% netn.Layers(3, 1).Name = net.Layers(3, 1).Name;
% netn.Layers(3, 1).InputSize = net.Layers(3, 1).InputSize;
% netn.Layers(3, 1).OutputSize = net.Layers(3, 1).OutputSize;
netn.Layers(3, 1).Weights = net.Layers(3, 1).Weights;
netn.Layers(3, 1).Bias = net.Layers(3, 1).Bias;
% netn.Layers(3, 1).WeightLearnRateFactor = net.Layers(3, 1).WeightLearnRateFactor;
% netn.Layers(3, 1).WeightL2Factor = net.Layers(3, 1).WeightL2Factor;
% netn.Layers(3, 1).BiasLearnRateFactor = net.Layers(3, 1).BiasLearnRateFactor;
% netn.Layers(3, 1).BiasL2Factor = net.Layers(3, 1).BiasL2Factor;
% netn.Layers(4, 1).Name = net.Layers(4, 1).Name;
% netn.Layers(4, 1).ResponseNames = net.Layers(4, 1).ResponseNames;
% netn.Layers(4, 1).LossFunction = net.Layers(4, 1).LossFunction;
%% 
Wi=netn.Layers(2, 1).InputWeights(1:200);
Wf=netn.Layers(2, 1).InputWeights(201:400);
Wg=netn.Layers(2, 1).InputWeights(401:600);
Wo=netn.Layers(2, 1).InputWeights(601:800);
Ri=netn.Layers(2, 1).RecurrentWeights(1:200,:);
Rf=netn.Layers(2, 1).RecurrentWeights(201:400,:);
Rg=netn.Layers(2, 1).RecurrentWeights(401:600,:);
Ro=netn.Layers(2, 1).RecurrentWeights(601:800,:);
bi=netn.Layers(2, 1).Bias(1:200);
bf=netn.Layers(2, 1).Bias(201:400);
bg=netn.Layers(2, 1).Bias(401:600);
bo=netn.Layers(2, 1).Bias(601:800);

Wfc = net.Layers(3, 1).Weights;
bfc = net.Layers(3, 1).Bias;

h=zeros(200,1);
c=zeros(200,1);
%% 
for Step=1:numTimeStepsTrain
    x=XTrain(Step);
    h_=h;
    c_=c;
    i=sigmoid(Wi*x+Ri*h_+bi);
    f=sigmoid(Wf*x+Rf*h_+bf);
    g=tanh(Wg*x+Rg*h_+bg);
    o=sigmoid(Wo*x+Ro*h_+bo);
    c=f.*c_+i.*g;
    h=o.*tanh(c);
%     hfc=Wfc*h+bfc;
end

x=YTrain(end);
h_=h;
c_=c;
i=sigmoid(Wi*x+Ri*h_+bi);
f=sigmoid(Wf*x+Rf*h_+bf);
g=tanh(Wg*x+Rg*h_+bg);
o=sigmoid(Wo*x+Ro*h_+bo);
c=f.*c_+i.*g;
h=o.*tanh(c);
hfc=Wfc*h+bfc;
YPred(1)=hfc;

numTimeStepsTest = numel(XTest);
for Step = 2:numTimeStepsTest
    x=YPred(Step-1);
    h_=h;
    c_=c;
    i=sigmoid(Wi*x+Ri*h_+bi);
    f=sigmoid(Wf*x+Rf*h_+bf);
    g=tanh(Wg*x+Rg*h_+bg);
    o=sigmoid(Wo*x+Ro*h_+bo);
    c=f.*c_+i.*g;
    h=o.*tanh(c);
    hfc=Wfc*h+bfc;
    YPred(Step)=hfc;
end
%% 
% YPred = sig*YPred + mu;
% YTest = (YTest - mu) / sig;
%% 
rmse = sqrt(mean((YPred-YTest).^2));
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
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Predicted"])
ylabel("Cases")
title("Forecast with Updates")

subplot(2,1,2)
stem(YPred - YTest)
xlabel("Month")
ylabel("Error")
title("RMSE = " + rmse)