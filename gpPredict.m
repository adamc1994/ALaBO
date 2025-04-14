function [y_pred, y_cov] = gpPredict(model, x_pred)
% This function makes predictions based on fitted GP model
% model - the fitted GP model containing the following parameters:
% -- params_star - optimal parameters found by MLE
% -- X - a matrix of input data, each row is a sample
% -- y - a vector of response
% -- dim_qual - an index array of qualitative variables
% -- d_lv - the dimension of latent space, 1 or 2
% -- levels - array of levels of each qualitative variable
% x_pred - m by d, m samples and d dim


% without qualitative variables
if numel(fieldnames(model)) == 5
    
    phi = model.phi; X = model.X; y = model.y;
    
    n = size(X,1);
    R_xpredx = computeR(phi,x_pred, X);
    R_xx = computeR(phi, X, X);
    R_xpred = computeR(phi, x_pred, x_pred);
    
    mu = (ones(1,n)*(R_xx\ones(n,1)))\(ones(1,n)*(R_xx\y)); 
    sigma = 1/n*(y-mu)'*(R_xx\(y-mu));
    y_pred = mu + R_xpredx*(R_xx\(y-mu));
    y_cov = sigma*(R_xpred - R_xpredx*(R_xx\R_xpredx'));

% with qualitative variables
elseif numel(fieldnames(model)) == 10
    
    phi = model.phi; z = model.z; X = model.X; y = model.y;
    dim_qual = model.dim_qual; d_lv = model.d_lv; levels = model.levels; log_noise = model.log_noise;
    
    [n,~] = size(X);
    d_qual = length(dim_qual);
    phi = [phi, zeros(1, d_lv*d_qual)];
    noise = exp(log_noise);
    
    X1 = toLatent(X,dim_qual, z, d_lv, levels);
    x_pred1 = toLatent(x_pred, dim_qual, z,d_lv, levels);
    
    R_xpredx = computeR(phi,x_pred1, X1, noise);
    R_xx = computeR(phi, X1, X1, noise);
    R_xpred = computeR(phi, x_pred1, x_pred1, noise);
    
    mu = (ones(1,n)*(R_xx\ones(n,1)))\(ones(1,n)*(R_xx\y)); 
    sigma = 1/n*(y-mu)'*(R_xx\(y-mu));

    y_pred = mu + R_xpredx*(R_xx\(y-mu));
    y_cov = sigma*(R_xpred - R_xpredx*(R_xx\R_xpredx'));
end
end

function R = computeR(phi, X1, X2, noise)
% This function computes the correlation matrix btween X1 and X2
% phi - lengthscale parameters
% X1, X2 - n by d, n sample size, d dim
% correlation is calculated using squared exponential formula

[n1,d] = size(X1);
n2 = size(X2,1);

d2 = 0;
for i = 1:d
    % distance matrix multiplied by theta
    d2 = d2 + 10.^(phi(i)).*(repmat(X1(:,i),[1,n2])-repmat(X2(:,i)',[n1,1])).^2;
end

if n1 ~= n2
    d = sqrt(d2); % Euclidean distance
    R = (1 + sqrt(5) * d + (5/3) * d.^2) .* exp(-sqrt(5) * d); % Matern 5/2
    % R = exp(- d2); % squared exponential
else
    d = sqrt(d2); % Euclidean distance
    R = (1 + sqrt(5) * d + (5/3) * d.^2) .* exp(-sqrt(5) * d); % Matern 5/2
    R = R + noise*eye(n1);
    % R = exp(-d2) + noise*eye(n1); % squared exponential
end
end
