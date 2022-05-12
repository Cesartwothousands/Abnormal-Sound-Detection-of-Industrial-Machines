clear
tic

subplot(2,2,1)
y = Load('vn421.wav');
fs = 16000;
N= length(y);
set(gcf,'position',[0.1,0.1,1000,600])
% Time domain
y1 = y;
t = 0:10/(N-1):10;
plot(t,y1);
axis([0,10,-0.1,0.1])
title('阀门样本信号时域图','FontSize',16)
xlabel('时间 t/s');
ylabel('幅度');
grid on


subplot(2,2,2)
y = Load('pn113.wav');
fs = 16000;
N= length(y);
set(gcf,'position',[0.1,0.1,1000,600])
% Time domain
y1 = y;
t = 0:10/(N-1):10;
plot(t,y1);
axis([0,10,-0.1,0.1])
title('水泵样本信号时域图','FontSize',16)
xlabel('时间 t/s');
ylabel('幅度');
grid on

subplot(2,2,3)
y = Load('fn444.wav');
fs = 16000;
N= length(y);
set(gcf,'position',[0.1,0.1,1000,600])
% Time domain
y1 = y;
t = 0:10/(N-1):10;
plot(t,y1);
axis([0,10,-0.1,0.1])
title('风扇样本信号时域图','FontSize',16)
xlabel('时间 t/s');
ylabel('幅度');
grid on

subplot(2,2,4)
y = Load('sn576.wav');
fs = 16000;
N= length(y);
set(gcf,'position',[0.1,0.1,1000,600])
% Time domain
y1 = y;
t = 0:10/(N-1):10;
plot(t,y1);
axis([0,10,-0.1,0.1])
title('滑轨样本信号时域图','FontSize',16)
xlabel('时间 t/s');
ylabel('幅度');
grid on

saveas(1,'41Time domain','png')
%close(figure(1))



% End, output running time
toc
