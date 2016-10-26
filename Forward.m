function [S, OX, AX, OY, AY] = Forward(X,net)

W1 = net.W1;
W2 = net.W2;
W3 = net.W3;

bias1 = net.bias1;
bias2 = net.bias2;
bias3 = net.bias3;

OH = zeros(size(X,2),net.layers{2}.size, size(X,1));
AH = zeros(size(X,2),net.layers{2}.size, size(X,1));
S  = zeros(size(X,2),net.layers{2}.size, size(X,1));
OY = zeros(size(X,2),net.layers{3}.size, size(X,1));
AY = zeros(size(X,2),net.layers{3}.size, size(X,1));

for a = 1:1:size(X,2)
    for b = 1:1:size(X,1)
       if b == 1
           oh = (W1*X(b,a))+ bias1;
           ah = Sigmoid(oh);
           oy = (W3*ah) + bias3;
           ay = oy
           
           OH(a,:, b) = oh;
           AH(a,:, b) = ah;
           OY(a,:, b) = oy;
           AY(a,:, b) = ay
       else
           
       end
        
        
    end
end
end

