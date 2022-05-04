clear
tic


path = 'fan.00.normal.00000000.wav';

y = Load(path);
fs = 16000;
N= length(y);

set(gcf,'position',[0.1,0.1,500,400])
% Time domain
y1 = y;
t = 0:10/(N-1):10;
plot(t,y1);
axis([0,10,-0.1,0.1])
title('风扇样本信号时域图')
xlabel('时间 t/s');
ylabel('幅度');
grid on
saveas(1,'FTime domain','png')
close(figure(1))



% End, output running time
toc
