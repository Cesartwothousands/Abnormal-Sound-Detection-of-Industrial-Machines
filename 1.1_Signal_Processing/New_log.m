clear
tic
path = 'vavle.00.normal.00000000.wav';



y = Load(path);fs = 16000;N= length(y);
t = 0:10/(N-1):10;

subplot(2,2,1);
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

subplot(2,2,2);
set(gcf,'position',[0.1,0.1,1000,1000])
[stftt,f2,t2] = stft(y,fs);
stftt = abs(stftt);
f2 = abs(f2);
f2 = abs(2512*log10(f2/700+1));
pcolor(t2,f2,stftt);shading interp
%colorbar('off')
%axis([0,10,0,8000]);
title('短时傅里叶时频谱图')
xlabel('时间 t/s');
ylabel('频率 f/Hz');
grid on

subplot(2,2,3);
set(gcf,'position',[0.1,0.1,1000,1000])
[stft3,f3,t3] = stft(y,fs);
stft3 = abs(stft3);
stft3 = 2595*log10(1+stft3/700);
pcolor(t3,f3,stft3);shading interp
%colorbar('off')
axis([0,10,0,8000]);
title('短时傅里叶时频谱图')
xlabel('时间 t/s');
ylabel('频率 f/Hz');
grid on

subplot(2,2,4);

% End, output running time
toc