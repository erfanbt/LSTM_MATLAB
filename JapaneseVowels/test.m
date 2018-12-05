%%
clear;
load('JV.mat');
%%
% XTest=XTest(1:13*27);
% YTest=YTest(1:13*27);
% for j=0:12
%     for i=1:27
%         [~,c(i)]=size(XTest{j*27+i,1});
%     end
%     m_c(j+1)=max(c);
%     for i=1:27
%         XTest{j*27+i,1} = padarray(XTest{j*27+i,1}',m_c(j+1)-c(i),'post')';
%     end
% end
%%
netn.Layers(2, 1).InputWeights = net.Layers(2, 1).InputWeights;
netn.Layers(2, 1).RecurrentWeights = net.Layers(2, 1).RecurrentWeights;
netn.Layers(2, 1).Bias = net.Layers(2, 1).Bias;
netn.Layers(3, 1).Weights = net.Layers(3, 1).Weights;
netn.Layers(3, 1).Bias = net.Layers(3, 1).Bias;
%%
Wi=netn.Layers(2, 1).InputWeights(1:100,:);
Wf=netn.Layers(2, 1).InputWeights(101:200,:);
Wg=netn.Layers(2, 1).InputWeights(201:300,:);
Wo=netn.Layers(2, 1).InputWeights(301:400,:);
Ri=netn.Layers(2, 1).RecurrentWeights(1:100,:);
Rf=netn.Layers(2, 1).RecurrentWeights(101:200,:);
Rg=netn.Layers(2, 1).RecurrentWeights(201:300,:);
Ro=netn.Layers(2, 1).RecurrentWeights(301:400,:);
bi=netn.Layers(2, 1).Bias(1:100);
bf=netn.Layers(2, 1).Bias(101:200);
bg=netn.Layers(2, 1).Bias(201:300);
bo=netn.Layers(2, 1).Bias(301:400);

Wfc = net.Layers(3, 1).Weights;
bfc = net.Layers(3, 1).Bias;

h=zeros(100,1);
c=zeros(100,1);

clear YPred;
%%
for k=0:12
    for m=1:27
        for Step=1:m_c(k+1)
            X=XTest{k*27+m,1};
            x=X(:,Step);
            h_=h;
            c_=c;
            i=sigmoid(Wi*x+Ri*h_+bi);
            f=sigmoid(Wf*x+Rf*h_+bf);
            g=tanh(Wg*x+Rg*h_+bg);
            o=sigmoid(Wo*x+Ro*h_+bo);
            c=f.*c_+i.*g;
            h=o.*tanh(c);
            hfc=Wfc*h+bfc;
            Y(Step,:)=hfc;
        end
        h=zeros(100,1);
        c=zeros(100,1);
        YP(k*27+m,:)=Y(end,:);
    end
end
[M,YPred] = max(YP,[],2);
%%
acc = sum(YPred == double(YTest))./numel(double(YTest));