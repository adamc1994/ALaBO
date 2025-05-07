classdef LVBayesianOptimiser
    % LVBayesianOptimiser: Bayesian optimization for mixed continuous and categorical variables.
    % Utilizes latent variable representations for categorical variables.
    % Supports acquisition strategies such as EI (Expected Improvement) and AEI (Adaptive EI).
    
    properties
        % Raw and scaled input data
        X = [];            % Original input data
        Xscaled            % Scaled input data (continuous variables normalized)
        
        % Raw and scaled output data
        y = [];            % Objective function values
        yscaled            % Standardized objective values (zero-mean, unit variance)
        ymu                % Mean of original y (for reverting)
        ysigma             % Std of original y (for reverting)
        
        % Problem dimensionality
        nvars              % Total number of variables (quantitative + qualitative)
        nquant             % Number of quantitative (continuous) variables
        nqual              % Number of qualitative (categorical) variables
        
        bounds = [];       % Bounds of all variables [2 x nvars]
        dim_qual           % Index positions of categorical variables
        levels             % Number of levels for each categorical variable
        
        % Latent variable config
        d_lv = 2;          % Latent variable dimension (1 or 2)
        n_starts = 200;    % Number of multi-starts for GP hyperparameter optimization
        
        mdl                % Trained GP model
        acq_type           % Acquisition function type: 'EI' or 'AEI'
        ratio = 0.05;      % EI-specific exploration constant
        ymin               % Minimum observed (predicted) y value
        contextual         % Contextual factor used in AEI (from predictive variance)
    end

    methods
        function obj = LVBayesianOptimiser(acq_type, bounds, X, y, dim_qual, levels)
            % Constructor: Initializes the optimiser with input data and settings
            
            if nargin == 0
                return
            end
            
            % Store user inputs
            obj.acq_type = acq_type;
            obj.bounds = bounds;
            obj.X = X;
            obj.y = y;
            obj.dim_qual = dim_qual;
            obj.nqual = length(dim_qual);
            obj.nquant = size(X,2) - obj.nqual;
            obj.levels = levels;
            
            % Normalize target and scale input data
            obj = obj.scaleY();
            obj = obj.scaleX();
            obj.nvars = size(X,2);

            % Fit the initial GP model
            obj.mdl = gpFit(obj.Xscaled,obj.yscaled, obj.n_starts, ...
                            obj.dim_qual, obj.d_lv, obj.levels);
            
            % Identify minimum y value and get its predicted value
            [~, ind] = min(obj.y);
            [obj.ymin, ~] = gpPredict(obj.mdl, obj.Xscaled(ind,:));
            
            % Handle AEI-specific contextual adjustment
            if strcmp(obj.acq_type, 'AEI')
                npoints = 10000;
                
                % Random sampling of quantitative variables
                xquant = obj.bounds(1,1:obj.nquant) + ...
                         (obj.bounds(2,1:obj.nquant) - obj.bounds(1,1:obj.nquant)) .* ...
                         rand(npoints, obj.nquant);
                
                % Random sampling of categorical variables (integers)
                xqual = zeros(npoints, obj.nqual);
                for i = 1:obj.nqual
                    xqual(:,i) = randi([obj.bounds(1,obj.nquant+i), obj.bounds(2,obj.nquant+i)], npoints, 1);
                end
                
                x = [xquant, xqual];  % Combine inputs
                % Scale quantitative part
                x(:,1:obj.nquant) = (x(:,1:obj.nquant) - obj.bounds(1,1:obj.nquant)) ./ ...
                                    (obj.bounds(2,1:obj.nquant) - obj.bounds(1,1:obj.nquant));
                
                % Predict variance and compute contextual factor
                [~, y_cov] = gpPredict(obj.mdl, x);
                obj.contextual = abs(mean(diag(y_cov)) / obj.ymin);
            end
        end
        
        function obj = addData(obj, X, y)
            % Adds new data to the training set and updates the model

            if ~isempty(obj.mdl)
                obj.X = [obj.X; X];
                obj.y = [obj.y; y];
            end

            % Update scaling and retrain the GP model
            obj = obj.scaleY();
            obj = obj.scaleX();
            obj.mdl = gpFit(obj.Xscaled, obj.yscaled, obj.n_starts, ...
                            obj.dim_qual, obj.d_lv, obj.levels);

            % Update ymin
            [~, ind] = min(obj.y);
            [obj.ymin, ~] = gpPredict(obj.mdl, obj.Xscaled(ind,:));
            
            % AEI-specific update
            if strcmp(obj.acq_type, 'AEI')
                npoints = 10000;
                xquant = obj.bounds(1,1:obj.nquant) + ...
                         (obj.bounds(2,1:obj.nquant) - obj.bounds(1,1:obj.nquant)) .* ...
                         rand(npoints, obj.nquant);
                xqual = zeros(npoints, obj.nqual);
                for i = 1:obj.nqual
                    xqual(:,i) = randi([obj.bounds(1,obj.nquant+i), obj.bounds(2,obj.nquant+i)], npoints, 1);
                end
                x = [xquant, xqual];
                x(:,1:obj.nquant) = (x(:,1:obj.nquant) - obj.bounds(1,1:obj.nquant)) ./ ...
                                    (obj.bounds(2,1:obj.nquant) - obj.bounds(1,1:obj.nquant));
                [~, y_cov] = gpPredict(obj.mdl, x);
                obj.contextual = abs(mean(diag(y_cov)) / obj.ymin);
            end
        end

        function obj = scaleY(obj)
            % Standardizes y to zero mean and unit variance
            [obj.yscaled, obj.ymu, obj.ysigma] = zscore(obj.y);
        end
        
        function y = revertY(obj, yscaled)
            % Converts standardized y back to original scale
            y = yscaled .* obj.ysigma + obj.ymu;
        end
        
        function obj = scaleX(obj)
            % Normalizes quantitative inputs to [0,1]
            obj.Xscaled = obj.X;
            obj.Xscaled(:,1:obj.nquant) = (obj.X(:,1:obj.nquant) - obj.bounds(1,1:obj.nquant)) ./ ...
                                          (obj.bounds(2,1:obj.nquant) - obj.bounds(1,1:obj.nquant));
        end
        
        function X = revertX(obj, Xscaled)
            % Converts scaled inputs back to original scale
            X = Xscaled;
            X(:,1:obj.nquant) = Xscaled(:,1:obj.nquant) .* ...
                                (obj.bounds(2,1:obj.nquant) - obj.bounds(1,1:obj.nquant)) + ...
                                obj.bounds(1,1:obj.nquant);
        end
        
        function f = acquisition(obj, X)
            % Computes acquisition function value at input locations
            % Supports EI and AEI

            % Normalize input
            X(:,1:obj.nquant) = (X(:,1:obj.nquant) - obj.bounds(1,1:obj.nquant)) ./ ...
                                 (obj.bounds(2,1:obj.nquant) - obj.bounds(1,1:obj.nquant));
            
            [mu, y_cov] = gpPredict(obj.mdl, X);
            sd = sqrt(diag(y_cov));
            clearvars y_cov

            % Select acquisition constant
            if strcmp(obj.acq_type,'EI')
                c = obj.ratio;
            elseif strcmp(obj.acq_type,'AEI')
                c = obj.contextual;
            end

            z = (obj.ymin - mu - c) ./ sd;

            % Expected Improvement formula (negated for minimization)
            f = -((obj.ymin - mu - c).*normcdf(z) + sd.*normpdf(z));
        end
        
        function [next, fval] = suggest(obj, p)
            % Suggests next sampling point(s) by optimizing the acquisition function
            % If p is provided, suggests a batch of p points

            if nargin < 2
                % Single-point suggestion using genetic algorithm
                opts = optimoptions(@ga, ...
                    'PopulationSize', 150, ...
                    'MaxGenerations', 200, ...
                    'EliteCount', 10, ...
                    'FunctionTolerance', 1e-8);

                [next, fval] = ga(@(x)obj.acquisition(x), obj.nvars, [], [], [], [], ...
                                  obj.bounds(1,:), obj.bounds(2,:), [], [obj.dim_qual], opts);
            else
                % Batch suggestion
                if p > 4
                    warning('Batches greater than 4 points lead to significant impact on algorithm performance');
                end

                next = zeros(p, obj.nvars);
                fval = zeros(p, 1);
                
                % Temporary model to avoid data contamination during loop
                tempopt = LVBayesianOptimiser(obj.acq_type, obj.bounds, obj.X, obj.y, obj.dim_qual, obj.levels);

                for i = 1:p
                    [next(i,:), fval(i)] = tempopt.suggest();
                    
                    % Scale to internal format
                    next(i,1:obj.nquant) = (next(i,1:obj.nquant) - obj.bounds(1,1:obj.nquant)) ./ ...
                                           (obj.bounds(2,1:obj.nquant) - obj.bounds(1,1:obj.nquant));
                    
                    % Predict y for the new point
                    [ynext, ~] = gpPredict(tempopt.mdl, next(i,:));
                    next(i,:) = tempopt.revertX(next(i,:));
                    ynext = tempopt.revertY(ynext);
                    
                    % Update model with new point
                    tempopt = tempopt.addData(next(i,:), ynext);
                end
            end
        end
    end
end
