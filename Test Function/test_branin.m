clc 
clear

% Define problem parameters
nquant = 1;                      % Number of quantitative (continuous) variables
nqual = 1;                       % Number of qualitative (categorical) variables

% Define bounds for each variable: [min; max]
% First column: x1 (quantitative), Second column: x2 (qualitative)
bounds = [0,1;                   % x1 ∈ [0, 1]
          1,4];                  % x2 ∈ {1, 2, 3, 4} (categorical levels)

npoints = 8;                     % Total number of initial sample points
minPerCat = 2;                   % Minimum number of points per categorical level

% Calculate number of categorical levels for x2
nCats = bounds(2, nquant+1) - bounds(1, nquant+1) + 1;

% Ensure enough points to cover each categorical level
assert(npoints >= nCats * minPerCat, 'Not enough points to guarantee coverage');

%% Step 1: Generate categorical samples (x2)
% Ensure minimum coverage for each level
xqual = repelem((bounds(1,nquant+1):bounds(2,nquant+1))', minPerCat);

% Fill in the remaining points randomly while preserving categorical balance
remaining = npoints - length(xqual);
if remaining > 0
    xqual = [xqual; randsample(xqual, remaining)];
end

% Randomize the order of categorical variable values
xqual = xqual(randperm(npoints));

%% Step 2: Generate Latin Hypercube samples for continuous variable (x1)
xquant = lhsdesign(npoints, nquant);  % Generate LHS samples in [0,1]

% Scale each sample dimension to its respective bounds
for i = 1:nquant
    xquant(:,i) = bounds(1,i) + xquant(:,i) .* (bounds(2,i) - bounds(1,i));
end

%% Combine x1 (continuous) and x2 (categorical) into a single input matrix
X = [xquant, xqual];

% Evaluate objective function (discretized Branin) for all initial points
for i = 1:npoints
    y(i,:) = discretized_branin(X(i,1),X(i,2));
end

% Define categorical variable metadata for the optimizer
levels = [4];                  % Number of levels for x2
dim_qual = [2];                % Column index of categorical variable in X

% Initialize the Bayesian optimizer
optimiser = LVBayesianOptimiser('AEI', bounds, X, y, dim_qual, levels);

%% Optimization loop
for i = 1:22
    % Suggest next best point(s) to sample
    [next, fval] = optimiser.suggest();

    % Evaluate the objective at the suggested point(s)
    for j = 1:size(next,1)
        ynext(j,:) = discretized_branin(next(j,1),next(j,2));
    end

    % Update the optimizer with new data
    optimiser = optimiser.addData(next,ynext);
end

%% Plot the trained model

% Generate a dense grid of x1 values for predictions (for each category)
sample_x1 = linspace(0,1,100)';              % 100 points along x1
sample_x1 = repmat(sample_x1, 4, 1);         % Repeat for each x2 level

% Construct corresponding x2 values (repeated categories)
sample_x2 = [ones(100,1); 2*ones(100,1); 3*ones(100,1); 4*ones(100,1)];

% Combine into full sample input matrix
sample_X = [sample_x1, sample_x2];

% Predict the mean and variance of the GP model at these sample points
[ypred, ycov] = gpPredict(optimiser.mdl, sample_X);

% Convert predictions back to original scale if they were normalized
ypred = optimiser.revertY(ypred);

% Use consistent color map for categories
colors = lines(4);  % default colormap with 4 distinct colors

%% Plotting

figure;
hold on;

% Scatter plot of observed data points, color-coded by categorical level
for i = 1:4
    idx = optimiser.X(:,2) == i;
    scatter(optimiser.X(idx,1), optimiser.y(idx), ...
            36, colors(i,:), 'filled', 'HandleVisibility', 'off');
end

% Plot predicted curves for each categorical level
for i = 1:4
    idx = (i-1)*100 + (1:100);  % Index range for each category
    plot(sample_x1(idx), ypred(idx), '--', ...
         'Color', colors(i,:), 'DisplayName', sprintf('x2 = %d', i));
end

% Plot actual (ground-truth) discretized Branin function for each category
x1_vals = linspace(0, 1, 500);
for u_idx = 1:4
    y_actual = arrayfun(@(x) discretized_branin(x, u_idx), x1_vals);
    plot(x1_vals, y_actual, 'LineWidth', 1.5, ...
         'Color', colors(u_idx,:), 'HandleVisibility', 'off');
end

% Formatting the plot
xlim([0 1]);
ylim([0 3]);
xlabel('x1');
ylabel('Branin Function Value');
legend;
grid on;
