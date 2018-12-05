%%
fileID = fopen('result.txt','w');

for WL=9
    clearvars -except WL fileID;
    load('JV_pad.mat');
    FL=WL-3;
    T = numerictype('WordLength',WL,'FractionLength',FL);
    T.Signed = true;
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
    %%
    for k=0:12
        for m=1:27
            for Step=1:m_c(k+1)
                X=XTest{k*27+m,1};
                X = fi(X,'numerictype',T);
                x=X(:,Step);
                h_=h;
                c_=c;
                i=sigmoid_fp(fi(fi(Wi*x,'numerictype',T)+fi(Ri*h_,'numerictype',T)+bi,'numerictype',T),WL,FL);
                f=sigmoid_fp(fi(fi(Wf*x,'numerictype',T)+fi(Rf*h_,'numerictype',T)+bf,'numerictype',T),WL,FL);
                g=tanh_fp(fi(fi(Wg*x,'numerictype',T)+fi(Rg*h_,'numerictype',T)+bg,'numerictype',T),WL,FL);
                o=sigmoid_fp(fi(fi(Wo*x,'numerictype',T)+fi(Ro*h_,'numerictype',T)+bo,'numerictype',T),WL,FL);
                c=fi(f.*c_,'numerictype',T)+fi(i.*g,'numerictype',T);
                c = fi(c,'numerictype',T);
                h=o.*tanh_fp(c,WL,FL);
                h = fi(h,'numerictype',T);
                hfc=fi(Wfc*h,'numerictype',T)+bfc;
                hfc = fi(hfc,'numerictype',T);
                Y(Step,:)=hfc;
            end
            h=zeros(100,1);
            c=zeros(100,1);
            h = fi(h,'numerictype',T);
            c = fi(c,'numerictype',T);
            YP(k*27+m,:)=Y(end,:);
        end
    end
    [M,YPred] = max(YP,[],2);
    %%
    acc = sum(YPred == double(YTest))./numel(double(YTest));
    fprintf(fileID, 'WL = %d\nFL = %d\n',WL,FL);
    fprintf(fileID, 'Accuracy = %f\n\n',acc);
end
fclose(fileID);