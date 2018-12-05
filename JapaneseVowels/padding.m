XTest=XTest(1:13*27);
YTest=YTest(1:13*27);
for j=0:12
    for i=1:27
        [~,c(i)]=size(XTest{j*27+i,1});
    end
    m_c(j+1)=max(c);
    for i=1:27
        XTest{j*27+i,1} = padarray(XTest{j*27+i,1}',m_c(j+1)-c(i),'post')';
    end
end

