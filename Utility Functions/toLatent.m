%% Utility function adapted from:
% "A Latent Variable Approach to Gaussian Process Modeling with Qualitative 
% and Quantitative Factors"
% https://www.tandfonline.com/doi/abs/10.1080/00401706.2019.1638834
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function X_latent = toLatent(X, dim_qual, z, d_lv, levels)
% Transforms input matrix X by replacing qualitative variables with their
% corresponding latent variable representations.
%
% Inputs:
%   X         - (n x d) matrix of input data (rows = samples)
%   dim_qual  - indices of qualitative variables in X
%   z         - vector of latent variables for qualitative levels
%   d_lv      - dimension of latent space per qualitative variable (1 or 2)
%   levels    - number of levels for each qualitative variable
%
% Output:
%   X_latent  - transformed input matrix with continuous and latent variables

[n,d] = size(X);
n_qual = length(dim_qual); % number of qualitative variables

% Preallocate latent representation matrix for qualitative variables
Z_qual = zeros(n, d_lv*n_qual);

% Loop through each qualitative variable
for i = 1:n_qual
    x_qual = X(:,dim_qual(i));  % Extract qualitative column

    % Number of latent variables already assigned (for indexing into z)
    if i ==1
        n_prev = 0;
    else
        n_prev = sum(levels(1:i-1)) - (i-1);
    end

    % Encode the first level with zero (as the base/reference)
    if d_lv==2
        Z_qual(x_qual==1,[2*i-1,2*i])=0;
    elseif d_lv==1
        Z_qual(x_qual==1,i)=0;
    end

    % Assign latent variable values for each additional level
    for j = 2:max(levels(i))
        if d_lv==2
            idx = [2*n_prev + 2*(j-1)-1, 2*n_prev + 2*(j-1)];
            Z_qual(x_qual==j,[2*i-1,2*i]) = repmat(z(idx), sum(x_qual==j), 1);
        elseif d_lv==1
            idx = n_prev + j - 1;
            Z_qual(x_qual==j,i) = repmat(z(idx), sum(x_qual==j), 1);
        end
    end
end

% Combine quantitative variables and newly created latent variables
if d > n_qual
    X_latent = [X(:,1:d-n_qual), Z_qual];
else
    X_latent = Z_qual;
end
end
