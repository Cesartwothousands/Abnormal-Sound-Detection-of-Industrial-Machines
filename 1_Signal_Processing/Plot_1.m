clear
tic

path = 'vavle.00.normal.00000023.wav';
P(path);


% End, output running time
toc

function P(p)
    name = p(1:1);
    y = Load(p);
    fs = 16000;
    N= length(y);
    
    set(gcf,'position',[0.1,0.1,1000,400])
    % Time domain
    y1 = y;
    t = 0:10/(N-1):10;
    plot(t,y1);
    axis([0,10,-0.1,0.1])
    title('样本信号时域图')
    xlabel('时间 t/s');
    ylabel('幅度');
    grid on
    saveas(1,[name,'T'],'png')
    close(figure(1))
    
    set(gcf,'position',[0.1,0.1,1000,400])
    % Frequency domain
    y2 = abs(fft(y,N));
    f1 = (0:N-1)*fs/N;
    %对数：y2 = 20*log10(y2);
    plot(f1,y2);
    axis([0,16000,0,20*log10(50)])
    title('样本信号频谱图')
    xlabel('频率 f/Hz');
    ylabel('幅度');
    grid on
    saveas(1,[name,'F'],'png')
    close(figure(1))
end


