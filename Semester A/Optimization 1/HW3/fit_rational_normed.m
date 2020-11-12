function [c0,c1,c2,d0,d1,d2] = fit_rational_normed(X)
    % get number of points in data
    m = numel(X(:,1));
    
    % extract x and y vectors
    xs = X(:,1);
    ys = X(:,2);
    
    % build A matrix 
    A = [ones(m,1),xs,xs.^2,-ys,-xs.*ys,-(xs.^2).*ys];
    
    % get eigen values and eigen vectors of A^TA
    [U,~,~] = svd(A'*A);
    
    % get the normlized eigen vector correponding to the minimal eigen value
    u = num2cell(U(:,6)./norm(U(:,6)));
    [c0,c1,c2,d0,d1,d2] = deal(u{:});
    
    
end

