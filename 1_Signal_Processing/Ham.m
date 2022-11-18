clear all;
%haming 
ham=hamming(60);
plot(ham,'b','lineWidth',0.75);
title('汉明窗时域波形');
xlabel('样点')
ylabel('幅度');
set(gca,'Ylim',[0,1.2])% $y\in[1,100]$
legend('Hamming')
grid on    
saveas(1,'Hamming','png')
close(figure(1))
