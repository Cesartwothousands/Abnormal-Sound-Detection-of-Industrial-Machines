clear
tic

path = 'vavle.00.normal.00000000.wav';

y = Load(path);
fs = 16000;
N= length(y);

set(gcf,'position',[0.1,0.1,1000,1000])
t = 0:10/(N-1):10;
stft(y,fs);
%colorbar('off')
axis([0,10,0,8])
title('短时傅里叶时频谱图')
xlabel('时间 t/s');
ylabel('频率 f/Hz');
grid on
saveas(1,'STFT','png')
close(figure(1))

% End, output running time
toc