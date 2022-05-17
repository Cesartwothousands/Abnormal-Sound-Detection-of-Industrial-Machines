t = 1:1:8000;
y1 = 2595*log10(1+t/700); 
y2 = log10(t);
%plot(y1,t,y2,t);
plot(t,y1);