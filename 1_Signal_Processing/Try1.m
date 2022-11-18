clear
tic

path = 'vavle.00.normal.00000023.wav';

y = Load(path);
y = 100*y;
fs = 16000;
dt=1/fs;    %时间精度
N= length(y);

set(gcf,'position',[0.1,0.1,1000,800])
t = 0:10/(N-1):10;
[wt,f] = cwt(y,'amor',fs);
pcolor(t,f,abs(wt));shading interp
title('连续小波变换时频谱图')
xlabel('时间 t/s');
ylabel('频率 f/Hz');
grid on
saveas(1,'sCWT','png')
close(figure(1))

% End, output running time
toc
