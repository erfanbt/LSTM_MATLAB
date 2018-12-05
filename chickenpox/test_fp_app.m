%%
% originalFormat = get(0, 'format');
% format loose
% format long g
% % Capture the current state of and reset the fi display and logging
% % preferences to the factory settings.
% fiprefAtStartOfThisExample = get(fipref);
% reset(fipref);
%
clear;
load('net_chickenpox.mat');
WL=16;
FL=10;
T = numerictype('WordLength',WL,'FractionLength',FL);
T.Signed = true;
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
Wi = fi(Wi,'numerictype',T);
Wf = fi(Wf,'numerictype',T);
Wg = fi(Wg,'numerictype',T);
Wo = fi(Wo,'numerictype',T);
Ri = fi(Ri,'numerictype',T);
Rf = fi(Rf,'numerictype',T);
Rg = fi(Rg,'numerictype',T);
Ro = fi(Ro,'numerictype',T);
bi = fi(bi,'numerictype',T);
bf = fi(bf,'numerictype',T);
bg = fi(bg,'numerictype',T);
bo = fi(bo,'numerictype',T);

Wfc = fi(Wfc,'numerictype',T);
bfc = fi(bfc,'numerictype',T);

h = fi(h,'numerictype',T);
c = fi(c,'numerictype',T);

XTrain = fi(XTrain,'numerictype',T);
YTrain = fi(YTrain,'numerictype',T);
XTest = fi(XTest,'numerictype',T);
%%
for Step=1:numTimeStepsTrain
    x=XTrain(Step);
    h_=h;
    c_=c;
    i=sigmoid_fp(fi(matvec_asmult(Wi,x,WL,FL)+matvec_asmult(Ri,h_,WL,FL)+bi,'numerictype',T));
    f=sigmoid_fp(fi(matvec_asmult(Wf,x,WL,FL)+matvec_asmult(Rf,h_,WL,FL)+bf,'numerictype',T));
    g=tanh_fp(fi(matvec_asmult(Wg,x,WL,FL)+matvec_asmult(Rg,h_,WL,FL)+bg,'numerictype',T));
    o=sigmoid_fp(fi(matvec_asmult(Wo,x,WL,FL)+matvec_asmult(Ro,h_,WL,FL)+bo,'numerictype',T));
    c=dotp_asmult(f,c_,WL,FL)'+dotp_asmult(i,g,WL,FL)';
    c = fi(c,'numerictype',T);
    h=dotp_asmult(o,tanh_fp(c),WL,FL)';
    h = fi(h,'numerictype',T);
    %     hfc=matvec_asmult(Wfc,h,WL,FL)+bfc;
    %     hfc = fi(hfc,'numerictype',T);
end

x=YTrain(end);
h_=h;
c_=c;
i=sigmoid_fp(fi(matvec_asmult(Wi,x,WL,FL)+matvec_asmult(Ri,h_,WL,FL)+bi,'numerictype',T));
f=sigmoid_fp(fi(matvec_asmult(Wf,x,WL,FL)+matvec_asmult(Rf,h_,WL,FL)+bf,'numerictype',T));
g=tanh_fp(fi(matvec_asmult(Wg,x,WL,FL)+matvec_asmult(Rg,h_,WL,FL)+bg,'numerictype',T));
o=sigmoid_fp(fi(matvec_asmult(Wo,x,WL,FL)+matvec_asmult(Ro,h_,WL,FL)+bo,'numerictype',T));
c=dotp_asmult(f,c_,WL,FL)'+dotp_asmult(i,g,WL,FL)';
c = fi(c,'numerictype',T);
h=dotp_asmult(o,tanh_fp(c),WL,FL)';
h = fi(h,'numerictype',T);
hfc=matvec_asmult(Wfc,h,WL,FL)+bfc;
hfc = fi(hfc,'numerictype',T);
YPred(1)=hfc;

numTimeStepsTest = numel(XTest);
for Step = 2:numTimeStepsTest
    x=YPred(Step-1);
    h_=h;
    c_=c;
    i=sigmoid_fp(fi(matvec_asmult(Wi,x,WL,FL)+matvec_asmult(Ri,h_,WL,FL)+bi,'numerictype',T));
    f=sigmoid_fp(fi(matvec_asmult(Wf,x,WL,FL)+matvec_asmult(Rf,h_,WL,FL)+bf,'numerictype',T));
    g=tanh_fp(fi(matvec_asmult(Wg,x,WL,FL)+matvec_asmult(Rg,h_,WL,FL)+bg,'numerictype',T));
    o=sigmoid_fp(fi(matvec_asmult(Wo,x,WL,FL)+matvec_asmult(Ro,h_,WL,FL)+bo,'numerictype',T));
    c=dotp_asmult(f,c_,WL,FL)'+dotp_asmult(i,g,WL,FL)';
    c = fi(c,'numerictype',T);
    h=dotp_asmult(o,tanh_fp(c),WL,FL)';
    h = fi(h,'numerictype',T);
    hfc=matvec_asmult(Wfc,h,WL,FL)+bfc;
    hfc = fi(hfc,'numerictype',T);
    YPred(Step)=hfc;
end
%%
rmse = sqrt(mean((double(YPred)-YTest).^2));
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
stem(double(YPred)-YTest)
xlabel("Month")
ylabel("Error")
title("RMSE = " + rmse)
