clear
tic
path = 'vavle.00.normal.00000000.wav';

y = Load(path);fs = 16000;N= length(y);
t = 0:10/(N-1):10;

subplot(1,4,1);
set(gcf,'position',[0.1,0.1,1000,1000])
stft(y,fs);
colorbar('off')
axis([0,10,0,8]);
title('短时傅里叶时频谱图');
xlabel('时间 t/s');
ylabel('频率 f/Hz');
grid on

subplot(1,4,2);
set(gcf,'position',[0.1,0.1,1000,1000])
melSpectrogram(y,fs);
colorbar('off')
%axis([0,10,0,8])
title('短时傅里叶时频谱图')
xlabel('时间 t/s');
ylabel('频率 f/Hz');
grid on

subplot(1,4,3);
set(gcf,'position',[0.1,0.1,1000,1000])
[stft1,f1,t1] = stft(y,fs);
stft1 = abs(stft1);
pcolor(t1,f1,stft1);shading interp
%colorbar('off')
axis([0,10,0,8000]);
title('短时傅里叶时频谱图')
xlabel('时间 t/s');
ylabel('频率 f/Hz');
grid on

subplot(1,4,4);
set(gcf,'position',[0.1,0.1,1000,1000])
[stftt,f2,t2] = stft(y,fs);
stftt = log(abs(stftt));
pcolor(t2,f2,stftt);shading interp
%colorbar('off')
axis([0,10,0,8000]);
title('短时傅里叶时频谱图')
xlabel('时间 t/s');
ylabel('频率 f/Hz');
grid on

% End, output running time
toc