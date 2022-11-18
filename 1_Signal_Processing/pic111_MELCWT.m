%clear
tic


subplot(2,2,1)
y = Load('vn421.wav');
fs = 16000;
N= length(y);
t = 0:10/(N-1):10;
set(gcf,'position',[0.1,0.1,800,500])
[wt,f] = cwt(y,'amor',fs);
wt1 = abs(wt);
pcolor(t,f,wt1);shading interp
%axis([0,10,0,8000]);
title('连续小波变换后的时频谱图')
xlabel('时间 t/s');
ylabel('频率 f(Hz)');
grid on

subplot(2,2,2)
y = Load('vn421.wav');
fs = 16000;
N= length(y);
t = 0:10/(N-1):10;
set(gcf,'position',[0.1,0.1,800,500])
[wt,f] = cwt(y,'amor',fs);
wt1 = abs(wt);
f = 2595*log10(1+f/700);
pcolor(t,f,wt1);shading interp
%axis([0,10,0,8000]);
title('联合梅尔标度后的时频谱图')
xlabel('时间 t/s');
ylabel('频率 f(mel)');
grid on


subplot(2,2,3)
y = Load('pn113.wav');
fs = 16000;
N= length(y);
t = 0:10/(N-1):10;
set(gcf,'position',[0.1,0.1,800,500])
[wt,f] = cwt(y,'amor',fs);
wt1 = abs(wt);
test1 = wt1;
pcolor(t,f,wt1);shading interp
%axis([0,10,0,8000]);
title('连续小波变换后的时频谱图')
xlabel('时间 t/s');
ylabel('频率 f(Hz)');
grid on


subplot(2,2,4)
y = Load('pn113.wav');
fs = 16000;
N= length(y);
t = 0:10/(N-1):10;
set(gcf,'position',[0.1,0.1,800,500])
[wt,f] = cwt(y,'amor',fs);
wt1 = abs(wt);
f = 2595*log10(1+f/700);
pcolor(t,f,wt1);shading interp
%axis([0,10,0,8000]);
title('联合梅尔标度后的时频谱图')
xlabel('时间 t/s');
ylabel('频率 f(mel)');
grid on

saveas(1,'画图Mel对比','png')
%close(figure(1))

% End, output running time
toc
