# 🧮 Micrograd in MATLAB

A MATLAB porting of [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd) — a tiny autograd engine and neural network library built from scratch for educational purposes.

## What is this?

This project reimplements micrograd entirely in MATLAB, providing a scalar-valued autograd engine with reverse-mode autodifferentiation (backpropagation) and a small neural network library on top of it. It follows Karpathy's famous [micrograd lecture](https://www.youtube.com/watch?v=VMj-3S1tku0) step by step.

The engine can compute gradients through arbitrary mathematical expressions, which is enough to train small neural networks.

## Core Components

| File | Description |
|------|-------------|
| `MicroValue.m` | The autograd engine. A scalar value node that tracks data, gradient, children, and the operation that produced it. Supports `+`, `*`, `^`, `-`, `/`, `exp`, and `tanh` with automatic gradient computation. |
| `MicroNeuron.m` | A single neuron: `tanh(w · x + b)` with randomly initialized weights and bias. |
| `MicroLayer.m` | A layer of neurons. |
| `MicroMLP.m` | A multi-layer perceptron built from stacked layers. |
| `micrograd1.m` | Part 1 of the lecture: numerical derivatives, building expression graphs, manual backpropagation, and a single neuron example. |
| `micrograd2.m` | Part 2 of the lecture: decomposing tanh into exp operations, gradient verification, building an MLP, and training it on a tiny dataset. |

## Quick Start

1. Clone the repo and open MATLAB in the project directory.
2. Run the first lecture script:
   ```matlab
   micrograd1
   ```
3. Run the second lecture script (builds and trains an MLP):
   ```matlab
   micrograd2
   ```

## Example: Training a Small Neural Network

```matlab
% Create a 3-input MLP with two hidden layers of 4 neurons and 1 output
net = MicroMLP(3, [4, 4, 1]);

% Training data
xs = { {2.0, 3.0, -1.0}, {3.0, -1.0, 0.5}, {0.5, 1.0, 1.0}, {1.0, 1.0, -1.0} };
ys = [1.0, -1.0, -1.0, 1.0];

% Training loop
for k = 1:20
    % Forward pass
    ypred = cell(1, numel(xs));
    for i = 1:numel(xs)
        ypred{i} = net.call(xs{i});
    end

    % Compute loss
    loss = MicroValue(0.0);
    for i = 1:numel(ys)
        loss = loss + (ypred{i} - ys(i)) ^ 2;
    end

    % Zero gradients, backward pass, update
    params = net.parameters();
    for i = 1:numel(params), params{i}.grad = 0.0; end
    loss.backward();
    for i = 1:numel(params)
        params{i}.data = params{i}.data - 0.1 * params{i}.grad;
    end
end
```

## Design Notes

- All classes use **handle semantics** so that object references are shared (similar to Python objects), which is essential for the computational graph to work correctly.
- `MicroValue.wrap()` automatically converts plain MATLAB numbers into `MicroValue` nodes, so you can write natural expressions like `node + 1` or `2 * node`.
- The backpropagation uses topological sorting to ensure gradients are computed in the correct order.

## License

MIT
