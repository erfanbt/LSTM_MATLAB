function [y] = matvec_asmult(W,x,WL,FL)

[r,~]=size(W);

for i=1:r
    y(i,1)=sum(dotp_asmult(W(i,:),x,WL,FL));
end

end