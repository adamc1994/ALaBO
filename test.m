clc 
clear
nquant = 1;
nqual = 1;
bounds = [0,1;
          1,2];
npoints = 10;
xquant = bounds(1,1:nquant) + ...
    (bounds(2,1:nquant)-bounds(1,1:nquant)).*...
    rand(npoints,nquant);
xqual = zeros(npoints,nqual);
for i = 1:nqual
    xqual(:,i) = randi(...
        [bounds(1,nquant+i),...
        bounds(2,nquant+i)],npoints,1);
end
X = [xquant,xqual];
y = -ftrig(X);
levels = [2];
dim_qual = [2];
optimiser = LVBayesianOptimiser('AEI', bounds, X, y, dim_qual, levels);
for i = 1:10
    [next, fval] = optimiser.suggest();
    ynext = -ftrig(next);
    optimiser = optimiser.addData(next,ynext);
end
