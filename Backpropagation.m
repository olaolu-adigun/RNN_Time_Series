function [delW1, delW2, delW3, del_bias1, del_bias2, del_bias3] = Backpropagation(OH, AH, OY, AY, ax, t, net)

%---Compute change for W3
ER = (t' - AY);
delW3 = ER * AH';
del_bias3 = sum(t' - AY);
 
%---Compute change for W2
delS = Sigmoid_derivative(AH);
delS_S  = zeros(size(OH,1), 1, size(OH,2) - 1);
delS_W2 = zeros(size(OH,1), size(net.W2,1),size(OH,2) - 1);
for  i = 2:1:size(ax,1)
    delS_S(:,:,i-1) = net.W2   * delS(:,i); 
    delS_W2(:,:, i-1)= delS(:,i)* (delS(:,i-1))';
end
 
delW2 = zeros(size(net.W2));
for j = 2:1:size(ax,1)
    er = ER(j) * 1 * net.W3;
    DEL = 1;
    for l = size(AH,2):-1:j
        gh  = delS_S( :, :, l-1);
        DEL = DEL.* gh;
    end
    pl = delS_W2(:,:, j-1);
    delW2 =  delW2 + (er' * DEL' *pl);
end



end

