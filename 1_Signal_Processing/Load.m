function l = load(path)  
%load sound
[y,Fs] = audioread(path);
l = 0;

for i = 1:8
    l = l + y(:,i);
end

l = l/8;

end
