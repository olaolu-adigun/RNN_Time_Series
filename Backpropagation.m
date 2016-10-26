function [delW1, delW2, delW3, del_bias1, del_bias2, del_bias3] = Backpropagation(OH, AH, OY, AY, ax, t, net)

%---Compute change for W3
 delW3 = (t' - AY) * AH';
 del_bias3 = sum(t' - AY);
 
 %---Compute change for W2
 
 for  i = 1:1:size(ax,1)
     
     
     
 end




end

