% this help us to build neural networks piece by piece. 

classdef MicroNeuron < handle % I use handle semantics so that references are shared (like Python objects).
% MICRONEURON  A single neuron: tanh(w . x + b)
% MATLAB porting of the Neuron class from Karpathy's micrograd.

    properties
        w   cell   % cell array of MicroValue (weights)
        b          % MicroValue (bias)
    end

    methods
        function obj = MicroNeuron(nin) % nin is the number of input entering the neuron
            obj.w = cell(1, nin);
            for i = 1:nin
                obj.w{i} = MicroValue(-1 + 2*rand());  % I inizialize weights randomicaly...
            end
            obj.b = MicroValue(-1 + 2*rand()); % ...and also the bias are inizialized randomicaly
        end

        function out = call(obj, x)
            % x is a cell array of MicroValue (or plain doubles)
            % Compute  w . x + b
            act = obj.b;
            for i = 1:numel(obj.w)
                act = act + obj.w{i} * MicroValue.wrap(x{i});
            end
            out = act.tanh(); % non linearity
        end

        function p = parameters(obj)
            p = [obj.w, {obj.b}];
        end
    end
end