%%
fileID = fopen('result_pwl_fp.txt','w');

for seg=32:32
    clearvars -except seg fileID;
    load('JV_pad_str.mat');
    WL=13;
    FL=WL-4;
    T = numerictype('WordLength',WL,'FractionLength',FL);
    T.Signed = true;
    %%    
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
                i=sigmoid_pwl_fp(fi(fi(Wi*x,'numerictype',T)+fi(Ri*h_,'numerictype',T)+bi,'numerictype',T),WL,FL,seg);
                f=sigmoid_pwl_fp(fi(fi(Wf*x,'numerictype',T)+fi(Rf*h_,'numerictype',T)+bf,'numerictype',T),WL,FL,seg);
                g=tanh_pwl_fp(fi(fi(Wg*x,'numerictype',T)+fi(Rg*h_,'numerictype',T)+bg,'numerictype',T),WL,FL,seg);
                o=sigmoid_pwl_fp(fi(fi(Wo*x,'numerictype',T)+fi(Ro*h_,'numerictype',T)+bo,'numerictype',T),WL,FL,seg);
                c=fi(f.*c_,'numerictype',T)+fi(i.*g,'numerictype',T);
                c = fi(c,'numerictype',T);
                h=o.*tanh_pwl_fp(c,WL,FL,seg);
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
    fprintf(fileID, 'WL = %d\nFL = %d\nseg = %d\n',WL,FL,seg);
    fprintf(fileID, 'Accuracy = %f\n\n',acc);
end
fclose(fileID);