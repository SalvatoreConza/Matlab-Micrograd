classdef MicroMLP < handle
% MICROMLP  Multi-Layer Perceptron.
%   MATLAB port of the MLP class from Karpathy's micrograd.

    properties
        layers  cell   % cell array of MicroLayer
    end

    methods
        function obj = MicroMLP(nin, nouts) % nouts è il numero di neuroni in uscita cioè nel layer successivo
            % nouts is a vector, e.g. [4, 4, 1]
            sz = [nin, nouts];
            obj.layers = cell(1, numel(nouts));
            for i = 1:numel(nouts)
                obj.layers{i} = MicroLayer(sz(i), sz(i+1));
            end
        end

        function out = call(obj, x)
            % x is a cell array of MicroValue (or doubles)
            % Ensure inputs are wrapped
            if ~iscell(x)
                error('Input x must be a cell array');
            end
            for i = 1:numel(obj.layers)
                x = obj.layers{i}.call(x);
                % If a single value came back, wrap in cell for next layer
                if ~iscell(x)
                    x = {x};
                end
            end
            % Unwrap if single output
            if iscell(x) && numel(x) == 1
                out = x{1};
            else
                out = x;
            end
        end

        function p = parameters(obj)
            p = {};
            for i = 1:numel(obj.layers)
                p = [p, obj.layers{i}.parameters()]; %#ok<AGROW>
            end
        end
    end
end