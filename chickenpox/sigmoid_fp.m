function [y] = sigmoid_fp(x)
y=1./(1+exp(-double(x)));
T = numerictype('WordLength',10,'FractionLength',7);
T.Signed = true;
y = fi(y,'numerictype',T);
end

