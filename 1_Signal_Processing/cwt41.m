clear
tic

subplot(2,2,1)
y = Load('vn421.wav');
y = 100*y;
fs = 16000;
dt=1/fs;    %时间精度
N= length(y);
set(gcf,'position',[0.1,0.1,1000,800])
t = 0:10/(N-1):10;
[wt,f] = cwt(y,'amor',fs);
pcolor(t,f,abs(wt));shading interp
%axis([0,8000,0,20*log10(50)])
title('阀门样本信号时频谱图','FontSize',16)
xlabel('时间 t/s');
ylabel('频率 f/Hz');
grid on

wcf=centfrq('amor'); %小波的中心频率
scal=fs*wcf./f;%利用频率转换尺度
coefs = cwt(y,scal,'amor');
figure(2)
pcolor(t,f,abs(coefs));shading interp


subplot(2,2,2)
y = Load('pn113.wav');
fs = 16000;
N= length(y);
set(gcf,'position',[0.1,0.1,1000,600])
% Frequency domain
y2 = abs(fft(y,N));
f1 = (0:N-1)*fs/N;
%对数：y2 = 20*log10(y2);
plot(f1,y2);
axis([0,8000,0,20*log10(50)])
title('水泵样本信号时频谱图','FontSize',16)
xlabel('时间 t/s');
ylabel('频率 f/Hz');
grid on

subplot(2,2,3)
y = Load('fn444.wav');
fs = 16000;
N= length(y);
set(gcf,'position',[0.1,0.1,1000,600])
% Frequency domain
y2 = abs(fft(y,N));
f1 = (0:N-1)*fs/N;
%对数：y2 = 20*log10(y2);
plot(f1,y2);
axis([0,8000,0,20*log10(50)])
title('风扇样本信号时频谱图','FontSize',16)
xlabel('时间 t/s');
ylabel('频率 f/Hz');
grid on

subplot(2,2,4)
y = Load('sn576.wav');
fs = 16000;
N= length(y);
set(gcf,'position',[0.1,0.1,1000,600])
% Frequency domain
y2 = abs(fft(y,N));
f1 = (0:N-1)*fs/N;
%对数：y2 = 20*log10(y2);
plot(f1,y2);
axis([0,8000,0,20*log10(50)])
title('滑轨样本信号频谱图','FontSize',16)
xlabel('时间 t/s');
ylabel('频率 f/Hz');
grid on

saveas(1,'411Fre domain','png')
%close(figure(1))



% End, output running time
toc
