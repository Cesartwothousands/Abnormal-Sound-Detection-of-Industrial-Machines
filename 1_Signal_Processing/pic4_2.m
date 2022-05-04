clear
tic

path = 'pump.00.normal.00000000.wav';

y = Load(path);
fs = 16000;
N= length(y);

set(gcf,'position',[0.1,0.1,1000,800])
t = 0:10/(N-1):10;
stft(y,fs);
colorbar('off')
axis([0,10,0,8])
title('短时傅里叶时频谱图')
xlabel('时间 t/s');
ylabel('频率 f/Hz');
grid on
saveas(1,'PSTFT','png')
close(figure(1))


set(gcf,'position',[0.1,0.1,1000,800])
t = 0:10/(N-1):10;
fsst(y,fs,kaiser(256,20),'yaxis')
colorbar('off')
%axis([0,10,-8,8])
title('连续小波变换时频谱图')
xlabel('时间 t/s');
ylabel('频率 f/Hz');
grid on
saveas(1,'PWSST','png')
close(figure(1))

set(gcf,'position',[0.1,0.1,1000,800])
t = 0:10/(N-1):10;
wsst(y,fs,'bump')
colorbar('off')
%axis([0,10,-8,8])
title('同步挤压小波变换时频谱图')
xlabel('时间 t/s');
ylabel('频率 f/Hz');
grid on
saveas(1,'PWWSST','png')
%close(figure(1))


% End, output running time
toc