function [OH, AH, OY, AY] = Forward(ax,net)

W1 = net.W1;
W2 = net.W2;
W3 = net.W3;

bias1 = net.bias1;
bias2 = net.bias2;
bias3 = net.bias3;

OH = zeros(net.layers{2}.size, size(ax,1));
AH = zeros(net.layers{2}.size, size(ax,1));
%S = zeros(net.layers{2}.size, size(ax,1));
OY = zeros(net.layers{3}.size, size(ax,1));
AY = zeros(net.layers{3}.size, size(ax,1));

for b = 1:1:size(ax,1)
    if b == 1
        oh = (W1*ax(1))+ bias1;
        ah = Sigmoid(oh);
        oy = (W3*ah) + bias3;
        ay = oy;
           
        OH(:, b) = oh;
        AH(:, b) = ah;
        OY(:, b) = oy;
        AY(:, b) = ay;
    else
        oh = (W1*ax(b))+ bias1;
        s  = AH(:,b-1);
        ah = Sigmoid(oh + ((W2*s)+ bias2));
        oy = (W3*ah) + bias3;
        ay = oy;
   
        OH(:, b) = oh;
        AH(:, b) = ah;
        OY(:, b) = oy;
        AY(:, b) = ay;
    end
end

end

