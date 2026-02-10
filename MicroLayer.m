classdef MicroLayer < handle % I use handle semantics so that references are shared (like Python objects)
% MICROLAYER  A layer of neurons.
%   MATLAB port of the Layer class from Karpathy's micrograd.

    properties
        neurons  cell   % cell array of MicroNeuron
    end

    methods
        function obj = MicroLayer(nin, nout)
            obj.neurons = cell(1, nout);
            for i = 1:nout
                obj.neurons{i} = MicroNeuron(nin);
            end
        end

        function out = call(obj, x)
            % x is a cell array of MicroValue
            outs = cell(1, numel(obj.neurons));
            for i = 1:numel(obj.neurons)
                outs{i} = obj.neurons{i}.call(x);
            end
            if numel(outs) == 1
                out = outs{1};       % single neuron => return scalar
            else
                out = outs;          % multiple => return cell array
            end
        end

        function p = parameters(obj)
            p = {};
            for i = 1:numel(obj.neurons)
                p = [p, obj.neurons{i}.parameters()]; %#ok<AGROW>
            end
        end
    end
end