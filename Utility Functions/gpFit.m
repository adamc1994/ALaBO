%% Utility function adapted from:
% "A Latent Variable Approach to Gaussian Process Modeling with Qualitative 
% and Quantitative Factors"
% https://www.tandfonline.com/doi/abs/10.1080/00401706.2019.1638834
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function model = gpFit(X, y, n_starts, dim_qual, d_lv, levels)
% Fits a Gaussian Process model with both qualitative and quantitative
% variables using maximum likelihood estimation.
%
% Inputs:
%   X         - (n x d) input matrix (qualitative vars to the right)
%   y         - (n x 1) response vector
%   n_starts  - number of random restarts for optimization
%   dim_qual  - indices of qualitative variables
%   d_lv      - latent space dimensionality (1 or 2)
%   levels    - number of levels per qualitative variable
%
% Output:
%   model     - struct containing fitted model parameters

if nargin == 6
    d = size(X,2); % total number of inputs
    n_quan = d - length(dim_qual); % number of quantitative variables

    % Count levels for each qualitative variable
    n_levels = zeros(length(dim_qual),1);
    for i = 1:length(dim_qual)
        n_levels(i) = length(unique(X(:,dim_qual(i))));
    end

    % Number of latent variables = (levels - 1)*dim for each qualitative var
    n_lv = d_lv * (sum(n_levels) - length(dim_qual));

    % Initial guesses
    phi0 = zeros(1, n_quan);
    z0 = -0.5 + rand(1, n_lv);
    log_noise0 = -10;
    params0 = [phi0, z0, log_noise0];

    % Define bounds for optimization
    if d_lv == 1
        lb = -2 * ones(1, n_quan);
        ub = 2 * ones(1, n_quan);
        for i = 1:length(dim_qual)
            lb = [lb, -3, -2 * ones(1, n_levels(i)-2)];
            ub = [ub, 3, 2 * ones(1, n_levels(i)-2)];
        end
    elseif d_lv == 2
        lb = -10 * ones(1, n_quan);
        ub = 10 * ones(1, n_quan);
        for i = 1:length(dim_qual)
            lb = [lb, -10, 0, -10 * ones(1, 2*n_levels(i)-4)];
            ub = [ub, 10, 0, 10 * ones(1, 2*n_levels(i)-4)];
        end
    end

    lb = [lb, -20]; % lower bound on log noise
    ub = [ub, 5];   % upper bound on log noise

    % Optimization setup
    options = optimoptions('fmincon','SpecifyObjectiveGradient',false);
    problem = createOptimProblem('fmincon',...
        'objective', @(params) logLikelihood(params,X,y,dim_qual,d_lv,levels),...
        'x0', params0, 'lb', lb, 'ub', ub, 'options', options);
    ms = MultiStart('UseParallel', true);

    % Run multi-start optimization
    [params_star, f] = run(ms, problem, n_starts);

    % Store fitted model
    model.X = X; model.y = y; model.n_starts = n_starts;
    model.dim_qual = dim_qual; model.d_lv = d_lv; model.levels = levels;
    model.phi = params_star(1:n_quan);
    model.z = params_star(n_quan+1 : n_quan+n_lv);
    model.log_noise = params_star(end);
    model.logL = f;
end
end
