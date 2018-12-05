function [y] = sigmoid_fp(x,WL,FL)
y=1./(1+exp(-double(x)));
T = numerictype('WordLength',WL,'FractionLength',FL);
T.Signed = true;
y = fi(y,'numerictype',T);
end

