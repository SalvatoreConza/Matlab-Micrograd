%% micrograd_lecture_part2.m
%  MATLAB porting of Andrej Karpathy's micrograd lecture (second half).
%  Requires: MicroValue.m, MicroNeuron.m, MicroLayer.m, MicroMLP.m
%  in the same folder / on the MATLAB path.

%% Setup
clear; clc; close all;
fprintf('=== Micrograd in MATLAB — Part 2 ===\n\n');

%% ---------------------------------------------------------------
%  Breaking tanh into exp operations 
%  Instead of using the built-in tanh, compute it manually as:
%     e = exp(2n),   o = (e - 1) / (e + 1)
%  ---------------------------------------------------------------
fprintf('=== tanh via exp decomposition ===\n');

% inputs x1, x2
x1 = MicroValue(2.0,  {}, '', 'x1');
x2 = MicroValue(0.0,  {}, '', 'x2');
% weights w1, w2
w1 = MicroValue(-3.0, {}, '', 'w1');
w2 = MicroValue(1.0,  {}, '', 'w2');
% bias
b  = MicroValue(6.8813735870195432, {}, '', 'b');

% x1*w1 + x2*w2 + b
x1w1 = x1 * w1;           x1w1.label = "x1*w1";
x2w2 = x2 * w2;           x2w2.label = "x2*w2";
x1w1x2w2 = x1w1 + x2w2;  x1w1x2w2.label = "x1w1+x2w2";
n = x1w1x2w2 + b;         n.label = "n";

% ---- manual tanh via exp ----
two_n = 2 * n;
e = two_n.exp();         e.label = "e";     
o = (e - 1) / (e + 1);    o.label = "o";
% ---- end manual tanh ----

o.backward();

fprintf('o.data = %.4f  (should be ~0.7071)\n', o.data);
o.draw();

%% ---------------------------------------------------------------
%  PyTorch comparison  
%  MATLAB doesn't have PyTorch, but we can verify our gradients
%  match the expected values from the lecture.
%  ---------------------------------------------------------------
fprintf('\n=== Gradient verification (compare with PyTorch) ===\n');
fprintf('Expected values from the lecture:\n');
fprintf('  o.data  = 0.7071\n');
fprintf('  x2.grad = 0.5000\n');
fprintf('  w2.grad = 0.0000\n');
fprintf('  x1.grad = -1.5000\n');
fprintf('  w1.grad = 1.0000\n');
fprintf('\nOur MicroValue results:\n');
fprintf('  o.data  = %.4f\n', o.data);
fprintf('  x2.grad = %.4f\n', x2.grad);
fprintf('  w2.grad = %.4f\n', w2.grad);
fprintf('  x1.grad = %.4f\n', x1.grad);
fprintf('  w1.grad = %.4f\n', w1.grad);

%% ---------------------------------------------------------------
%  Neural network classes: Neuron, Layer, MLP 
%  See MicroNeuron.m, MicroLayer.m, MicroMLP.m
%  ---------------------------------------------------------------
fprintf('\n=== Building an MLP ===\n');

%% Single forward pass 
rng(42);   % fix seed for reproducibility (like random.seed in Python)
x_input = {2.0, 3.0, -1.0};       % cell array = Python list
net = MicroMLP(3, [4, 4, 1]);      % 3 inputs -> 4 -> 4 -> 1

y_single = net.call(x_input);
fprintf('Single forward pass output: %.4f\n', y_single.data);

%% Training data 
xs = {
    {2.0,  3.0, -1.0}
    {3.0, -1.0,  0.5}
    {0.5,  1.0,  1.0}
    {1.0,  1.0, -1.0}
};
ys = [1.0, -1.0, -1.0, 1.0];   % desired targets

%% Training loop for 20 steps
fprintf('\n=== Training loop (20 iterations) ===\n');
fprintf('%4s  %12s\n', 'Step', 'Loss');
fprintf('%s\n', repmat('-', 1, 18));

for k = 1:20

    % --- forward pass ---
    ypred = cell(1, numel(xs));
    for i = 1:numel(xs)
        ypred{i} = net.call(xs{i});
    end

    % loss = sum( (yout - ygt)^2 )
    loss = MicroValue(0.0);
    for i = 1:numel(ys)
        diff_i = ypred{i} - ys(i);     % MicroValue - scalar
        loss = loss + diff_i ^ 2;
    end

    % --- zero gradients ---
    params = net.parameters();
    for i = 1:numel(params)
        params{i}.grad = 0.0;
    end

    % --- backward pass ---
    loss.backward();

    % --- gradient descent update ---
    for i = 1:numel(params)
        params{i}.data = params{i}.data + (-0.1) * params{i}.grad; % params{i}.data is the loss function to diminuish and -0.1 is the learning rate
    end

    fprintf('%4d  %12.6f\n', k, loss.data);
end

%% Show final predictions
fprintf('\n=== Final predictions ===\n');
for i = 1:numel(xs)
    yp = net.call(xs{i});
    fprintf('  xs{%d} -> %.4f   (target: %+.1f)\n', i, yp.data, ys(i));
end

fprintf('\nDone!\n');
