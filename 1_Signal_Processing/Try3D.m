clear
tic



y = Load('sn576.wav');
fs = 16000;
N= length(y);
t = 0:10/(N-1):10;
%set(gcf,'position',[0.1,0.1,800,500])

stft(y,fs,'Window',kaiser(256,5),'OverlapLength',220,'FFTLength',512)

axis([0,10,0,8,-50,0]);
title('阀门样本信号时频谱图')
xlabel('时间 t/s');
ylabel('频率 f/kHz');
zlabel('幅度');
grid on

view(-45,65)
colormap jet

%saveas(1,'41STFT','png')
%close(figure(1))

% End, output running time
toc
