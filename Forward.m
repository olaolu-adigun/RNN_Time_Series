function [S, OX, AX, OY, AY] = Forward(X,net)

W1 = net.W1;
W2 = net.W2;
W3 = net.W3;

bias1 = net.bias1;
bias2 = net.bias2;
bias3 = net.bias3;

S = zeros(size(X'));
for a = 1:1:size(s,1)
    for b = 1:1:size(X,1)
       if b == 1
           ox = W*X(b,a);
           ax = Sigmoid(ox);
           
       else
           
       end
        
        
    end
end
end

