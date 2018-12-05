function [y] = tanh_fp(x)
y=tanh(double(x));
T = numerictype('WordLength',10,'FractionLength',7);
T.Signed = true;
y = fi(y,'numerictype',T);
end

