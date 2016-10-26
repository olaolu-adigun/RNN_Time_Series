
clear;
[X,T] = exchanger_dataset;
X = X'; T = T';

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

M = 5;
assert(size(train_x,1)==size(train_y, 1), 'Check the data set.');

%% Initialize optimization parameters and weights  

%---Optimization parameter

opts.numepochs = 100;
opts.batch = size(train_x,2) - M + 1;
opts.learning = 0.1;

%---Define the Neural Network
nn.layers = {
    struct('type', 'input', 'size', I)
    struct('type', 'hidden', 'size', J)
    struct('type', 'output','size', K)
    };

%---Weight and bias random range
e = 0.3;
b = -e;

%% Initialize the weights 
nn.W1 = unifrnd(b, e,nn.layers{2}.size, nn.layers{1}.size);
nn.W2 = unifrnd(b, e,nn.layers{2}.size, nn.layers{2}.size);
nn.W3 = unifrnd(b, e,nn.layers{3}.size, nn.layers{2}.size);


% Initiialize Bias Weight 
nn.Bias_W1 = unifrnd(b, e, nn.layers{2}.size,1);
nn.Bias_W2 = unifrnd(b, e, nn.layers{3}.size,1);

nnb.Bias_W1 = unifrnd(b, e, nn.layers{2}.size,1);
nnb.Bias_W2 = nn.Bias_W2;

nnb.Bias_W1 = nn.Bias_W1;

