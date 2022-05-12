clear
tic

path = 'vavle.00.normal.00000023.wav';

y = Load(path);
fs = 16000;
N= length(y);

set(gcf,'position',[0.1,0.1,1000,600])
t = 0:10/(N-1):10;
cwt(y,fs);
%colorbar('off')
%axis([0,10,-8,8])
title('连续小波变换时频谱图')
xlabel('时间 t/s');
ylabel('频率 f/Hz');
grid on
saveas(1,'CWT','png')
% close(figure(1))

% End, output running time
toc