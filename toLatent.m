%% Utility functions taken from:
% A Latent Variable Approach to Gaussian Process Modeling with Qualitative 
% and Quantitative Factors
% https://www.tandfonline.com/doi/abs/10.1080/00401706.2019.1638834
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function X_latent = toLatent(X, dim_qual, z, d_lv, levels)

% This function transforms the original X into latent space
% X - a matrix of input data, each row is a sample
% dim_qual - an index array of qualitative variables
% z - an array of the latent variable values for qualitative variables
% d_lv - the dimension of latent space, 1 or 2
% levels - an array of the levels of each qualitative variable
% returns a matrix contains both continuous x variables and the latent variables z

% scale_factor = 10 / std(z(:));  % normalise spread if required
% z = z * scale_factor;

[n,d] = size(X);
n_qual = length(dim_qual); % number of qualitative variables

Z_qual = zeros(n, d_lv*n_qual); % matrix of latent variables

for i = 1:n_qual

    x_qual = X(:,dim_qual(i));
    
    if i ==1
        n_prev = 0;
    else
        n_prev = sum(levels(1:i-1)) - (i-1);
    end

    
    if d_lv==2
        Z_qual(x_qual==1,[2*i-1,2*i])=0; % set the first level to be zero in latent space
    elseif d_lv==1
        Z_qual(x_qual==1,i)=0;
    end
    
    
    % plug x_lv into each level of qualitative variables
    for j = 2:max(levels(i))
        if d_lv==2
            Z_qual(x_qual==j,[2*i-1,2*i])=repmat(z([2*n_prev + 2*(j-1)-1,2*n_prev + 2*(j-1)]),[sum(x_qual==j),1]);
        elseif d_lv==1
            Z_qual(x_qual==j,i)=repmat(z(n_prev + j-1),[sum(x_qual==j),1]);
        end
    end

end

% combine quantitative and qualitative variables
if d > n_qual
    X_latent = [X(:,1:d-n_qual), Z_qual];
else
    X_latent = Z_qual;
end
end
