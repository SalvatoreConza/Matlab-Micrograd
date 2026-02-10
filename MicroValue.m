classdef MicroValue < handle % I use handle semantics so that references are shared (like Python objects).
    % MicroValue. A scalar value node in a computational graph with autograd support.
    % Tracks data, gradient, children, and the operation that produced it.

    properties
        data      double  = 0.0 % the actual numerical value of this node
        grad      double  = 0.0 % the derivative of the loss function L with respect to this node
        label     string  = ""  % the name of this node, it can be a, b, c, d, e, f, g, e, chico ecc...
        op        string  = ""  % the operation that in this node is done, can be a multiplication, an addition, an application of a iperbolic tan ecc...
    end

    properties (Access = public)
        prev      % list of children (input nodes) that produce this node
        backward_fn  % this function is build to remember how to calculate the derivative for this specific operation 
        % es: if this is c and c, have as children a and b and it's operation is + then c is a+b and then dc/da=1 and dc/db=1
        % es: if this is f and f haves children d and e and it's operations is * then f is d*e a and then df/dd=e and df/de=d
    end

    methods
        %% constructor
        function obj = MicroValue(data, children, op, label)
            obj.data = data;
            obj.grad = 0.0; % gradient always start from zero
            obj.backward_fn = @() [];  % no-op by default
            % when I call a function I pass some arguments to it, nargin
            % allows me to pass less argument in the function
            if nargin >= 2
                obj.prev = children;
            else
                obj.prev = {};
            end
            if nargin >= 3
                obj.op = op;
            end
            if nargin >= 4
                obj.label = label;
            end
        end

        %% Display
        function disp(obj)
            fprintf('MicroValue(label=%s, data=%.4f, grad=%.4f)\n', obj.label, obj.data, obj.grad);
        end

        %% Addition: out = self + other
        % perform the actual math and create the new node for the addition
        % now wraps plain scalars automatically so you can do: node + 1
        function out = plus(self, other)
            other = MicroValue.wrap(other);
            self  = MicroValue.wrap(self);
            out = MicroValue(self.data + other.data, {self, other}, '+');
            % Capture references for the backward closure
            s = self;
            o = other;
            out.backward_fn = @() backward_add(s, o, out);
        end

        %% Multiplication: out = self * other
        % perform the actual math and create the new node for the multiplication
        % now wraps plain scalars automatically so you can do: 2 * node
        function out = mtimes(self, other)
            other = MicroValue.wrap(other);
            self  = MicroValue.wrap(self);
            out = MicroValue(self.data * other.data, {self, other}, '*');
            s = self;
            o = other;
            out.backward_fn = @() backward_mul(s, o, out);
        end

        %% Power: out = self ^ k   (k must be a plain number, not a MicroValue)
        % es: if out = self^k then d(out)/d(self) = k * self^(k-1)
        function out = mpower(self, k)
            assert(isnumeric(k), 'Only numeric powers supported for now');
            out = MicroValue(self.data ^ k, {self}, sprintf('**%g', k));
            s = self;
            out.backward_fn = @() backward_pow(s, k, out);
        end

        % .^ also works (element-wise power, same thing for scalars)
        function out = power(self, k)
            out = mpower(self, k);
        end

        %% Change of sign: out = -self
        % just multiply by -1, reuses the * backward
        function out = uminus(self)
            out = self * (-1);
        end

        %% Subtraction: out = self - other
        % reuses + and unary minus:  self + (-other)
        function out = minus(self, other)
            out = MicroValue.wrap(self) + (-MicroValue.wrap(other));
        end

        %% Division: out = self / other
        % reuses * and ^:  self * other^(-1)
        function out = mrdivide(self, other)
            out = MicroValue.wrap(self) * (MicroValue.wrap(other) ^ (-1));
        end

        %% Exp: out = exp(self)
        % es: if out = e^self then d(out)/d(self) = e^self = out.data
        function out = exp(self)
            e = builtin('exp', self.data);   % call MATLAB's built-in exp
            out = MicroValue(e, {self}, 'exp');
            s = self;
            out.backward_fn = @() backward_exp(s, out);
        end

        %% Tanh activation
        % perform the actual math and create the new node for the iperbolic tangent
        function out = tanh(self)
            t = (builtin('exp', 2*self.data) - 1) / (builtin('exp', 2*self.data) + 1);
            out = MicroValue(t, {self}, 'tanh');
            s = self;
            out.backward_fn = @() backward_tanh(s, t, out);
        end

        %% Full backward pass (topological sort + reverse-mode autodiff)
        function backward(self)
            % Build topological order, we need to make sure the backprop is done correctrly
            topo = {};
            visited = {};  % cell array of MicroValue handles we've seen

            function build_topo(v)
                % Check if v is already visited (compare handles)
                already = false;
                for k = 1:numel(visited)
                    if v == visited{k}
                        already = true;
                        break;
                    end
                end
                if ~already
                    visited{end+1} = v;
                    for k = 1:numel(v.prev)
                        build_topo(v.prev{k});
                    end
                    topo{end+1} = v;
                end
            end

            build_topo(self);

            % seed gradient
            self.grad = 1.0;

            % reverse pass
            for i = numel(topo):-1:1
                topo{i}.backward_fn();
            end
        end

        %% Graph visualisation (text-based)
        function draw(root)
            % trace all nodes and edges
            [nodes, ~] = trace_graph(root);
            fprintf('\n--- Computational Graph ---\n');
            for k = 1:numel(nodes)
                n = nodes{k};
                childLabels = "";
                for j = 1:numel(n.prev)
                    childLabels = childLabels + n.prev{j}.label + " ";
                end
                if n.op == ""
                    fprintf('  [%s]  data=%.4f  grad=%.4f  (input)\n', ...
                        n.label, n.data, n.grad);
                else
                    fprintf('  [%s]  data=%.4f  grad=%.4f  op=%s  children={%s}\n', ...
                        n.label, n.data, n.grad, n.op, strtrim(childLabels));
                end
            end
            fprintf('---------------------------\n\n');
        end
    end

    %% ---- Static helper to wrap plain numbers into MicroValue ----
    methods (Static)
        function v = wrap(x)
            % If x is already a MicroValue, return it as-is.
            % If x is a plain number, wrap it in a new MicroValue node.
            % This lets you write things like:  2 * node,  node + 1,  node - 3
            if isa(x, 'MicroValue')
                v = x;
            else
                v = MicroValue(x);
            end
        end
    end
end

%% ---- Local backward functions (defined outside the class) ----

function backward_add(self, other, out)
    self.grad  = self.grad  + 1.0 * out.grad;
    other.grad = other.grad + 1.0 * out.grad;
end

function backward_mul(self, other, out)
    self.grad  = self.grad  + other.data * out.grad;
    other.grad = other.grad + self.data  * out.grad;
end

function backward_pow(self, k, out)
    % d(self^k)/d(self) = k * self^(k-1)
    self.grad = self.grad + k * (self.data ^ (k - 1)) * out.grad;
end

function backward_exp(self, out)
    % d(exp(self))/d(self) = exp(self) = out.data
    self.grad = self.grad + out.data * out.grad;
end

function backward_tanh(self, t, out)
    self.grad = self.grad + (1 - t^2) * out.grad;
end

%% ---- Graph tracing helper ----

function [nodes, edges] = trace_graph(root)
    nodes = {};
    edges = {};

    function build(v)
        already = false;
        for k = 1:numel(nodes)
            if v == nodes{k}
                already = true;
                break;
            end
        end
        if ~already
            nodes{end+1} = v;
            for k = 1:numel(v.prev)
                edges{end+1} = {v.prev{k}, v};
                build(v.prev{k});
            end
        end
    end

    build(root);
end