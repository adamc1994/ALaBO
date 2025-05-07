%% Utility function adapted from:
% "A Latent Variable Approach to Gaussian Process Modeling with Qualitative 
% and Quantitative Factors"
% https://www.tandfonline.com/doi/abs/10.1080/00401706.2019.1638834
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function logL = logLikelihood(params, X, y, dim_qual, d_lv, levels)
% Computes the negative log-likelihood of the Gaussian Process model.
%
% Inputs:
%   params    - vector of GP hyperparameters [phi, z, log_noise]
%   X         - (n x d) input matrix
%   y         - (n x 1) output vector
%   dim_qual  - indices of qualitative variables
%   d_lv      - latent space dimension for qualitative variables
%   levels    - levels for each qualitative variable
%
% Output:
%   logL      - negative log-likelihood

if nargin == 6
    [n,d] = size(X);
    d_qual = length(dim_qual); % number of qualitative vars

    % Separate hyperparameters
    z = params(d - d_qual + 1 : end - 1); % latent variables
    log_noise = params(end); % noise (log)
    noise = exp(log_noise);
    
    % Transform inputs to latent space
    X1 = toLatent(X, dim_qual, z, d_lv, levels);
    
    % Extract length-scale parameters and compute correlation matrix
    phi = params(1:d - d_qual);
    R = computeR([phi, zeros(1, d_lv*d_qual)], X1, X1, noise);

    % Cholesky decomposition of correlation matrix
    one_n = ones(1,n);
    L = chol(R,'lower');

    R_inv_y = L'\(L\y); 
    R_inv_one = L'\(L\one_n');

    % Compute GP mean
    mu = (one_n * R_inv_y) / (one_n * R_inv_one);
    R_inv_y_mu = L'\(L\(y - mu));

    % Compute GP variance and negative log-likelihood
    sigma2 = 1/n * (y - mu)' * R_inv_y_mu;
    logL = log(det(R)) + n * log(sigma2); % No constant term added
end
end
