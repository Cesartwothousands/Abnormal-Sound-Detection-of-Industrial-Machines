
set(gcf,'position',[0.1,0.1,1000,500])

y=[88.12  91.41;91.08  92.20;81.47 85.16;79.01 77.65;];
b=bar(y);
axis([0,5,70,100]);
grid on;
%ch = get(b,'children');
set(gca,'XTickLabel',{'阀门','水泵','风扇','滑轨'})


set(gca,'FontSize',20)
legend('短时傅里叶变换','优化时频变换','FontSize',16);
xlabel('工业机械类别');
ylabel('分类准确率 /%');
