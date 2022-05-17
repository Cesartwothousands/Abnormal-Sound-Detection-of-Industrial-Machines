clear
tic


subplot(2,2,1)
y = Load('vn421.wav');
fs = 16000;
N= length(y);
t = 0:10/(N-1):10;
set(gcf,'position',[0.1,0.1,800,500])
[stft1,f1,t1] = stft(y,fs);
stft1 =  abs(stft1);
f1 = log10(f1);
pcolor(t1,abs(f1),stft1);shading interp
axis([0,10,0,8000]);
title('阀门样本信号时频谱图')
xlabel('时间 t/s');
ylabel('频率 f/Hz');
grid on

subplot(2,2,2)
y = Load('pn113.wav');
fs = 16000;
N= length(y);
t = 0:10/(N-1):10;
set(gcf,'position',[0.1,0.1,800,500])
[stft1,f1,t1] = stft(y,fs);
stft1 =  abs(stft1);
pcolor(t1,f1,stft1);shading interp
axis([0,10,0,8000]);
title('水泵样本信号时频谱图')
xlabel('时间 t/s');
ylabel('频率 f/Hz');
grid on


subplot(2,2,3)
y = Load('fn444.wav');
fs = 16000;
N= length(y);
t = 0:10/(N-1):10;
set(gcf,'position',[0.1,0.1,800,500])
[stft1,f1,t1] = stft(y,fs);
stft1 =  abs(stft1);
test2 = stft1;
pcolor(t1,f1,stft1);shading interp
axis([0,10,0,8000]);
title('风扇样本信号时频谱图')
xlabel('时间 t/s');
ylabel('频率 f/Hz');
grid on


subplot(2,2,4)
y = Load('sn576.wav');
fs = 16000;
N= length(y);
t = 0:10/(N-1):10;
set(gcf,'position',[0.1,0.1,800,500])
[stft1,f1,t1] = stft(y,fs);
stft1 =  abs(stft1);
pcolor(t1,f1,stft1);shading interp
axis([0,10,0,8000]);
title('滑轨样本信号时频谱图')
xlabel('时间 t/s');
ylabel('频率 f/Hz');
grid on

saveas(1,'41STFT','png')
%close(figure(1))

% End, output running time
toc
