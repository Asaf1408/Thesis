[X,Y] = meshgrid(-2:0.1:2,-2:0.1:2);
Z = X.^2 - X.^2.*Y.^2 + Y.^4;
surf(X,Y,Z);