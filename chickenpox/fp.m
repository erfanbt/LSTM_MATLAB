% originalFormat = get(0, 'format');
% format loose
% format long g
% % Capture the current state of and reset the fi display and logging
% % preferences to the factory settings.
% fiprefAtStartOfThisExample = get(fipref);
% reset(fipref);

T = numerictype('WordLength',3,'FractionLength',2);
T.Signed = true;
a = fi(2,'numerictype',T);
b = double(a);
% c = fi(c,'numerictype',T);

% % Reset the fi display and logging preferences
% fipref(fiprefAtStartOfThisExample);
% set(0, 'format', originalFormat);