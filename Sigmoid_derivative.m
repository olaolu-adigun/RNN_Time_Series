function [ del_oh ] = Sigmoid_derivative(ah)

%%---Find the derivative of Sigmoid Function
del_oh = (1 - ah).* ah;

end
