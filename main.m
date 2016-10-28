clear;
[X,T] = exchanger_dataset;

X = cell2mat(X); 
T = cell2mat(T);

%---Training set
train_x = X(:,1:3000);
train_y = T(:,1:3000);

[Z , mu, sigma] = zscore(train_y);
train_y = Z;

%---Test set
test_x = X(:, 3001:4000);
test_y = T(:, 3001:4000);

test_y = (1/sigma)*(test_y - mu);

%---Size of Layers
I = size(train_x,1);
J = 6;
K = size(train_y,1);

M = 3;
assert(size(train_x,1)==size(train_y, 1), 'Check the data set.');

%% Initialize optimization parameters and weights  
%---Optimization parameter

opts.numepochs = 100;
opts.learning = 0.1;

%---Define the Neural Network
nn.layers = {
    struct('type', 'input', 'size', I)
    struct('type', 'hidden', 'size', J)
    struct('type', 'output','size', K)
    };

%---Weight and bias random range
e = 0.5;
b = -e;

%% Initialize the weights 
nn.W1 = unifrnd(b, e,nn.layers{2}.size, nn.layers{1}.size);
nn.W2 = unifrnd(b, e,nn.layers{2}.size, nn.layers{2}.size);
nn.W3 = unifrnd(b, e,nn.layers{3}.size, nn.layers{2}.size);

%---Initiialize Bias Weight 
nn.bias1 = unifrnd(b, e, nn.layers{2}.size,1);
nn.bias2 = unifrnd(b, e, nn.layers{2}.size,1);
nn.bias3 = unifrnd(b, e, nn.layers{3}.size,1);

%% PREPROCESSING DATA
num_train = size(train_x,2);
num_test =  size(test_x,2);

train_X = train_x;
train_Y = train_y;

test_X = test_x;
test_Y = test_y;

%% Training
% Feed-Forward Propagation
train_MSE = zeros(opts.numepochs,1);

for iter = 1:1:opts.numepochs
    opts.learning = opts.learning * (0.9999^iter);
    for j = 1:1:size(train_X, 2)
        ax = train_X(:,j);
        ay = train_Y(:,j);
        
        [OH, AH, OY, AY] = Forward(ax, nn);
        [delW1, delW2, delW3, del_bias1, del_bias2, del_bias3] = Backpropagation(OH, AH, OY,AY, ax, ay, nn);
    end
end
