%% Utility function adapted from:
% "A Latent Variable Approach to Gaussian Process Modeling with Qualitative 
% and Quantitative Factors"
% https://www.tandfonline.com/doi/abs/10.1080/00401706.2019.1638834
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function R = computeR(phi, X1, X2, noise)
% Computes the correlation matrix between two sets of inputs using the
% Matern 5/2 kernel.
%
% Inputs:
%   phi   - log-scale length parameters for each input dimension
%   X1    - (n1 x d) input matrix
%   X2    - (n2 x d) input matrix
%   noise - noise variance to be added on diagonal (only when X1 == X2)
%
% Output:
%   R     - (n1 x n2) correlation matrix

[n1,d] = size(X1);
n2 = size(X2,1);

% Compute squared, weighted Euclidean distances
d2 = 0;
for i = 1:d
    diff = repmat(X1(:,i), [1,n2]) - repmat(X2(:,i)', [n1,1]);
    d2 = d2 + 10.^(phi(i)) .* diff.^2;
end

% Matern 5/2 kernel with optional noise on diagonal
d = sqrt(d2);
R = (1 + sqrt(5)*d + (5/3)*d.^2) .* exp(-sqrt(5)*d);

if n1 == n2
    R = R + noise * eye(n1); % Add noise only for auto-covariance matrix
end
end
