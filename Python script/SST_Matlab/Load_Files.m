close all
clear all
clc

% 
% c = f1(1,1);
info =audioinfo('vavle.00.normal.00000023.wav');%获取音频文件的信息
[y,Fs] = audioread('vavle.00.normal.00000023.wav');
audiolength = length(y);%获取音频文件的数据长度
t = 1:1:audiolength;

figure(1),
plot(t,y(1:audiolength));
xlabel('Time');
ylabel('Audio Signal');
title('原始音频文件信号幅度图');
c = f1(1,1);
function y = f1(x1,x2)  
    y = x1 + x2;
end

