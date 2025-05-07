%% Utility function adapted from:
% "A Latent Variable Approach to Gaussian Process Modeling with Qualitative 
% and Quantitative Factors"
% https://www.tandfonline.com/doi/abs/10.1080/00401706.2019.1638834
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [y_pred, y_cov] = gpPredict(model, x_pred)
% Predicts GP mean and variance at new input locations.
%
% Inputs:
%   model   - GP model struct returned by gpFit()
%   x_pred  - (m x d) matrix of test inputs
%
% Outputs:
%   y_pred  - (m x 1) vector of predictive means
%   y_cov   - (m x m) predictive covariance matrix

if numel(fieldnames(model)) == 10
    % Extract model parameters
    phi = model.phi; z = model.z;
    X = model.X; y = model.y;
    dim_qual = model.dim_qual;
    d_lv = model.d_lv;
    levels = model.levels;
    log_noise = model.log_noise;

    [n,~] = size(X);
    d_qual = length(dim_qual);
    phi = [phi, zeros(1, d_lv*d_qual)];
    noise = exp(log_noise);

    % Transform training and prediction points to latent space
    X1 = toLatent(X, dim_qual, z, d_lv, levels);
    x_pred1 = toLatent(x_pred, dim_qual, z, d_lv, levels);

    % Compute kernel matrices
    R_xpredx = computeR(phi, x_pred1, X1, noise);
    R_xx = computeR(phi, X1, X1, noise);
    R_xpred = computeR(phi, x_pred1, x_pred1, noise);

    % Predictive mean and covariance
    mu = (ones(1,n)*(R_xx\ones(n,1))) \ (ones(1,n)*(R_xx\y)); 
    sigma = 1/n * (y - mu)' * (R_xx \ (y - mu));

    y_pred = mu + R_xpredx * (R_xx \ (y - mu));
    y_cov = sigma * (R_xpred - R_xpredx * (R_xx \ R_xpredx'));
end
end
