function [y] = tanh_pwl_fp(x)
% originalFormat = get(0, 'format');
% format loose
% format long g
% % Capture the current state of and reset the fi display and logging
% % preferences to the factory settings.
% fiprefAtStartOfThisEx(d)ample = get(fipref);
% reset(fipref);

T = numerictype('WordLength',8,'FractionLength',5);
T.Signed = true;

step=8/16;
for k=1:16
    a(k)=(tanh(-4+step*k)-tanh(-4+step*(k-1)))/step;
    b(k)=tanh(-4+step*k)-a(k)*(-4+step*k);
end
%
% sample = -2:.0001:2; 
% v = tanh(sample);
a = fi(a,'numerictype',T);
b = fi(b,'numerictype',T);
xq = -4:step:4;
% figure
% vq1 = interp1(sample,v,xq);
% plot(xq,vq1,':.');
% xlim([-2 2]);
% title('(Default) Linear Interpolation');
for d=1:length(x)
    for dd=1:length(xq)-1
        if((x(d)>= xq(dd)) &&(x(d) < xq(dd+1)))
            y(d)=a(dd)*x(d)+b(dd);
%             disp(dd);
        elseif(x(d)<= xq(1))
            y(d)=tanh(xq(1));
%             disp(dd);
        elseif(x(d)>= xq(end))
            y(d)=tanh(xq(end));
%             disp(dd);
        end
    end
end
%
% for d=1:length(x)
%     if (x(d) <= -2)
%         y(d)=tanh(-2);
%     elseif (-2 > x(d) >= -1.8)
%         y(d)=a(1)*x(d)+b(1);
%     elseif (-1.8 > x(d) >= -1.6)
%         y(d)=a(2)*x(d)+b(2);
%     elseif (-1.6 > x(d) >= -1.4)
%         y(d)=a(3)*x(d)+b(3);
%     elseif (-1.4 > x(d) >= -1.2)
%         y(d)=a(4)*x(d)+b(4);
%     elseif (-1.2 > x(d) >= -1)
%         y(d)=a(5)*x(d)+b(5);
%     elseif (-1 > x(d) >= -0.8)
%         y(d)=a(6)*x(d)+b(6);
%     elseif (-0.8 > x(d) >= -0.6)
%         y(d)=a(7)*x(d)+b(7);
%     elseif (-0.6 > x(d) >= -0.4)
%         y(d)=a(8)*x(d)+b(8);
%     elseif (-0.4 > x(d) >= -0.2)
%         y(d)=a(9)*x(d)+b(9);
%     elseif (-0.2 > x(d) >= 0)
%         y(d)=a(10)*x(d)+b(10);
%     elseif (0 < x(d) <= 0.2)
%         y(d)=a(11)*x(d)+b(11);
%     elseif (0.2 < x(d) <= 0.4)
%         y(d)=a(12)*x(d)+b(12);
%     elseif (0.4 < x(d) <= 0.6)
%         y(d)=a(13)*x(d)+b(13);
%     elseif (0.6 < x(d) <= 0.8)
%         y(d)=a(14)*x(d)+b(14);
%     elseif (0.8 < x(d) <= 1)
%         y(d)=a(15)*x(d)+b(15);
%     elseif (1 < x(d) <= 1.2)
%         y(d)=a(16)*x(d)+b(16);
%     elseif (1.2 < x(d) <= 1.4)
%         y(d)=a(17)*x(d)+b(17);
%     elseif (1.4 < x(d) <= 1.6)
%         y(d)=a(18)*x(d)+b(18);
%     elseif (1.6 < x(d) <= 1.8)
%         y(d)=a(19)*x(d)+b(19);
%     elseif (1.8 < x(d) <= 2)
%         y(d)=a(20)*x(d)+b(20);
%     else
%         y(d)=tanh(2);
%     end
% end
y=transpose(y);
y = fi(y,'numerictype',T);

% % Reset the fi display and logging preferences
% fipref(fiprefAtStartOfThisEx(d)ample);
% set(0, 'format', originalFormat);
end