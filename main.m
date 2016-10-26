
clear;
[X,T] = exchanger_dataset;

%---Training set
train_x = X(:,1:3000);
train_y = T(:,1:3000);

%---Test set
test_x = P(:, 3001:4000);
test_y = P(:, 3001:4000);

%---Size of Layers
I = size(train_x,1);
J = 6;
K = size(train_y,1);


