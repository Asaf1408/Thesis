function [c0,c1,c2,d1,d2] = fit_rational(X)
    % get number of points in data
    m = numel(X(:,1));
    
    % extract x and y vectors
    xs = X(:,1);
    ys = X(:,2);
    
    % build A matrix and b vector
    A = [ones(m,1),xs,xs.^2,-xs.*ys,-(xs.^2).*ys];
    b = ys;
    
    % solve least square problem
    u_tilde = num2cell(((A'*A)\(A'*b)));
    [c0,c1,c2,d1,d2] = deal(u_tilde{:});

end

