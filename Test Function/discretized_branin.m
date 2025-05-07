function y = discretized_branin(x1, u_idx)
    %DISCRETIZED_BRANIN Evaluates a discretized version of the Branin function.
    %
    %   y = DISCRETIZED_BRANIN(x1, u_idx) computes the Branin function value
    %   for a given normalized input x1 ∈ [0, 1] and a categorical index u_idx ∈ {1,2,3,4}.
    %
    %   The Branin function is evaluated after transforming the normalized x1
    %   to its original domain and selecting a fixed x2 value based on the
    %   categorical level u_idx. The result is returned in log10 scale.

    % Constants used in the standard Branin function
    b = 5 / (4 * pi^2);
    c = 5 / pi;
    r = 6;
    s = 10;
    t = 1 / (8 * pi);

    % Define the original input bounds for x1 and x2
    xmin = [-5; 0];     % Lower bounds
    xmax = [10; 15];    % Upper bounds

    % Define discretized levels for the categorical variable x2
    u = [0, 0.333, 0.666, 1];  % Normalized positions for x2 within its domain

    % Input validation: ensure the categorical index is valid
    if u_idx < 1 || u_idx > length(u)
        error('u_idx must be between 1 and 4.');
    end

    % Transform normalized x1 ∈ [0,1] back to its original domain [-5, 10]
    x1 = xmin(1) + (xmax(1) - xmin(1)) * x1;

    % Transform categorical level to fixed x2 value in original domain [0, 15]
    x2 = xmin(2) + (xmax(2) - xmin(2)) * u(u_idx);

    % Evaluate the Branin function at (x1, x2)
    y = (x2 - b * x1^2 + c * x1 - r)^2 + s * (1 - t) * cos(x1) + s;

    % Return the log-scaled function value for numerical stability and interpretability
    y = log10(y);
end
