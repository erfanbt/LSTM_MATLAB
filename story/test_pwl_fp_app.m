%%
fileID = fopen('results_pwl_app.txt','w');
seg=32;

for WL=21:25
    clearvars -except WL fileID seg;
    load('net_story_mod.mat');
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
        i=sigmoid_pwl_fp(fi(matvec_asmult(Wi,x,WL,FL)+matvec_asmult(Ri,h_,WL,FL)+bi,'numerictype',T),WL,FL,seg);
        f=sigmoid_pwl_fp(fi(matvec_asmult(Wf,x,WL,FL)+matvec_asmult(Rf,h_,WL,FL)+bf,'numerictype',T),WL,FL,seg);
        g=tanh_pwl_fp(fi(matvec_asmult(Wg,x,WL,FL)+matvec_asmult(Rg,h_,WL,FL)+bg,'numerictype',T),WL,FL,seg);
        o=sigmoid_pwl_fp(fi(matvec_asmult(Wo,x,WL,FL)+matvec_asmult(Ro,h_,WL,FL)+bo,'numerictype',T),WL,FL,seg);
        c=dotp_asmult(f,c_,WL,FL)'+dotp_asmult(i,g,WL,FL)';
        c = fi(c,'numerictype',T);
        h=dotp_asmult(o,tanh_pwl_fp(c,WL,FL,seg),WL,FL)';
        h = fi(h,'numerictype',T);
        hfc=matvec_asmult(Wfc,h,WL,FL)+bfc;
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