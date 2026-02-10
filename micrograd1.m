%% micrograd_lecture.m
%  MATLAB porting of Andrej Karpathy's micrograd lecture (first half).
%  Requires: Value.m (handle class) in the same folder.

%% Setup
clear; clc; close all; % removes variables, clear command window and close any plot 
fprintf('=== Micrograd in MATLAB ===\n\n');

%% Simple function and its evaluation in a certain point
f = @(x) 3*x.^2 - 4*x + 5;
fprintf('f(3.0) = %f\n', f(3.0));           % 20.0

%% Plot the function
xs = -5:0.25:4.75;
ys = f(xs);
figure; plot(xs, ys, 'LineWidth', 1.5);
title('f(x) = 3x^2 - 4x + 5'); xlabel('x'); ylabel('f(x)'); grid on;

%%  Numerical derivative at x = 2/3, formal definition
h = 1e-6;
x = 2/3;
numerical_deriv = (f(x + h) - f(x)) / h;
fprintf('Numerical derivative at x=2/3: %.6f  (analytic: 0.0)\n\n', numerical_deriv);

%% More complex expression & manual derivative check, look to Xournal in red
a_val = 2.0;
b_val = -3.0;
c_val = 10.0;
d_val = a_val*b_val + c_val;
fprintf('d = a*b + c = %.4f\n', d_val);

h = 0.0001;
a_val = 2.0; 
b_val = -3.0;

c_val = 10.0; % this is c 
d1 = a_val*b_val + c_val; % this is d(c)

c_val = c_val + h; % this is c+h
d2 = a_val*b_val + c_val; % this is d(c+h)
fprintf('d1=%.4f  d2=%.4f  slope(dd/dc)=%.4f\n\n', d1, d2, (d2-d1)/h);

%% Build a small expression graph with Value and backprop, look to Xournal in blue
a = MicroValue(2.0,  {}, '', 'a');
b = MicroValue(-3.0, {}, '', 'b');
c = MicroValue(10.0, {}, '', 'c');
e = a * b;      e.label = "e";
d = e + c;      d.label = "d";
fv = MicroValue(-2.0, {}, '', 'f');   % "fv" to avoid shadowing the function f
L = d * fv;     L.label = "L";

disp(L);

%% Draw the graph (text-based)
L.draw();

%% Numerical gradient check with Value objects
fprintf('--- Numerical gradient check (nudge b) ---\n');
h = 0.001;
% Forward pass 1
a1 = MicroValue(2.0); 
b1 = MicroValue(-3.0); 
c1 = MicroValue(10.0); 
f1 = MicroValue(-2.0);
e1 = a1*b1; 
d1v = e1+c1; 
L1 = d1v*f1;
L1_data = L1.data;

% Forward pass 2 (nudge b)
a2 = MicroValue(2.0);  
b2 = MicroValue(-3.0); 
b2.data = b2.data + h;
c2 = MicroValue(10.0);
f2 = MicroValue(-2.0);
e2 = a2*b2; 
d2v = e2+c2; 
L2 = d2v*f2;
L2_data = L2.data;

fprintf('dL/db (numerical) = %.4f\n\n', (L2_data - L1_data)/h);

%% Backprop on the original graph
L.backward();
fprintf('--- After L.backward() ---\n');
L.draw();

%% One step of gradient descent
a.data = a.data + 0.01 * a.grad;
b.data = b.data + 0.01 * b.grad;
c.data = c.data + 0.01 * c.grad;
fv.data = fv.data + 0.01 * fv.grad;

% Recompute forward pass
e = a * b;
d = e + c;
L_new = d * fv;
fprintf('L after one GD step = %.4f  (was -8.0000)\n\n', L_new.data);

%% Plot tanh
figure;
t_xs = -5:0.2:4.8;
plot(t_xs, tanh(t_xs), 'LineWidth', 1.5);
title('tanh(x)'); xlabel('x'); ylabel('tanh(x)'); grid on;

%% Single neuron forward + backward
fprintf('=== Single neuron example ===\n');
x1 = MicroValue(2.0,  {}, '', 'x1');
x2 = MicroValue(0.0,  {}, '', 'x2');
w1 = MicroValue(-3.0, {}, '', 'w1');
w2 = MicroValue(1.0,  {}, '', 'w2');
b  = MicroValue(6.8813735870195432, {}, '', 'b');

x1w1 = x1 * w1;   x1w1.label = "x1*w1";
x2w2 = x2 * w2;   x2w2.label = "x2*w2";
x1w1x2w2 = x1w1 + x2w2;  x1w1x2w2.label = "x1w1+x2w2";
n = x1w1x2w2 + b;         n.label = "n";
o = n.tanh();              o.label = "o";

fprintf('Before backward:\n');
o.draw();

o.backward();

fprintf('After backward:\n');
o.draw();

%% Verify dtanh/dn = 1 - o^2
fprintf('1 - o.data^2 = %.4f  (should equal n.grad = %.4f)\n\n', ...
    1 - o.data^2, n.grad);

%% a + a  (tests grad accumulation)
fprintf('=== Grad accumulation test (b = a + a) ===\n');
a = MicroValue(3.0, {}, '', 'a');
bv = a + a;  bv.label = "b";
bv.backward();
bv.draw();
fprintf('a.grad should be 2.0: %.4f\n\n', a.grad);

%% More complex: f = d * e, d = a*b, e = a+b
fprintf('=== f = (a*b) * (a+b) ===\n');
a = MicroValue(-2.0, {}, '', 'a');
bv = MicroValue(3.0, {}, '', 'b');
d = a * bv;    d.label = "d";
e = a + bv;    e.label = "e";
fv = d * e;    fv.label = "f";

fv.backward();
fv.draw();

fprintf('Done!\n');

