clear all;
close all;
clc;

[X,Y] = meshgrid(0.000001:0.01:10,0.000001:0.01:10);
Z = -(X.*log(X) + Y .*log(Y) - (X+Y) .* log(X+Y));
figure();
mesh(X,Y,Z);