function [y] = tanh_fp(x,WL,FL)
y=tanh(double(x));
T = numerictype('WordLength',WL,'FractionLength',FL);
T.Signed = true;
y = fi(y,'numerictype',T);
end

