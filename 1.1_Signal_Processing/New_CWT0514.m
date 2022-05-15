%clear
tic

path = 'vavle.00.normal.00000023.wav';

y = Load(path);
fs = 16000;
dt=1/fs;    %时间精度
N= length(y);

subplot(1,2,1);
set(gcf,'position',[0.1,0.1,800,500])
t = 0:10/(N-1):10;
[wt,f] = cwt(y,'amor',fs);
wt1 = abs(wt);
pcolor(t,f,wt1);shading interp
title('连续小波变换时频谱图');
xlabel('时间 t/s');
ylabel('频率 f/Hz');
grid on
%saveas(1,'sCWT','png')
%close(figure(1))

subplot(1,2,2);
set(gcf,'position',[0.1,0.1,800,500])
t = 0:10/(N-1):10;
[wt,f] = cwt(y,'amor',fs);
wt1 = abs(wt);
f = 2595*log10(1+f/700);
pcolor(t,f,wt1);
title('连续小波变换时频谱图');shading interp
xlabel('时间 t/s');
ylabel('频率 f/Hz');
grid on
saveas(1,'try','png')
%close(figure(1))

% End, output running time
toc
