


% 第四种 morlet小波函数
d=-4,h=4,n=100;
[g41,xval]=morlet(d,h,n);
figure
subplot(211)
plot(xval,g41,'r','LineWidth',0.75);
title('Morlet小波函数时域波形')
xlabel('时间t')
ylabel('幅度')
grid on
set(gcf,'position',[0.1,0.1,800,500])
g42=abs(fft(g41));
subplot(212);
plot(g42,'b');
title('Morlet小波函数幅频波形')
xlabel('频率f')
ylabel('幅度')
grid on
saveas(1,'Morlet','png')