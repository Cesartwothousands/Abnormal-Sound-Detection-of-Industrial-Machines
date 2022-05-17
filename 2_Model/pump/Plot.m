
acc = pumpacc{:,1};
loss = pumpacc{:,2};
vacc = pumpacc{:,3};
vloss = pumpacc{:,4};
t = 1:1:1000;
t =t';

set(gcf,'position',[0.1,0.1,1000,600])
plot(t,acc,'b',t,vacc,'r','lineWidth',1.25);
%plot(t,loss,'b',t,vloss,'r','lineWidth',1.25);
ylabel('准确率','FontSize',13)
xlabel('训练轮次','FontSize',13)   
axis([-20,1050,0.8,1.04]);
% set(gca,'XLim',[0,1050])% $x\in[1,100]$
% set(gca,'Ylim',[0.6,1.04])% $y\in[1,100]$
set(gca,'FontSize',18)
legend('训练集','验证集','FontSize',13)
grid on
