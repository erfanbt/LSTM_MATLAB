%%
fileID = fopen('results_pwl.txt','w');

for seg=32
    clearvars -except seg fileID;
    load('net_story_mod.mat');
    WL=15;
    FL=WL-3;
    T = numerictype('WordLength',WL,'FractionLength',FL);
    T.Signed = true;
    %%    
    h=zeros(200,1);
    c=zeros(200,1);
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
    
    X = fi(X,'numerictype',T);
    Y = fi(Y,'numerictype',T);
    %%
    for Step=1:111
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
        YPred(:,Step)=hfc;
    end
    %%
    Y=double(Y);
    YPred=double(YPred);
    for i=1:111
        n_YPred(i)=norm(YPred(:,i));
        n_Y(i)=norm(Y(:,i));
    end
    rmse = norm(YPred-Y);
    %%
    fprintf(fileID, 'WL = %d\nFL = %d\nseg = %d\n',WL,FL,seg);
    fprintf(fileID, 'RMSE = %f\n\n',rmse);
end
fclose(fileID);