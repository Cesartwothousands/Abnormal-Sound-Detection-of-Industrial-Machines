tic

y = Load('vn421.wav');
fs = 16000;
N= length(y);
t = 0:10/(N-1):10;
set(gcf,'position',[0.1,0.1,1000,1000])
[wt,f] = cwt(y,'amor',fs);
wt1 = abs(wt);
f = 2595*log10(1+f/700);
pcolor(t,f,wt1);shading interp
%axis([0,10,0,8000]);

axis off
grid on

saveas(1,'用来画图','png')

toc