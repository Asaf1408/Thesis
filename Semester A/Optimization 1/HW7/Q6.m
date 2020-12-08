%% cleaning the worksapce
clear all;
close all;
clc;

%% section a

Q = [1,0,0;0,2,-1;0,-1,3];
P = [4,-4,0;-4,6,0;0,0,0];
cvx_begin
variable x(3)
minimize max([norm([2,-3,0]*x,1),norm([-1,1,1]*x,1)]) + quad_form(x,Q)
subject to
    pow_pos(quad_form(x,P)+0.01,8) + quad_over_lin([0,0,1]*x,[2,3,0]*x) -150 <= 0;
    [-1,-1.5,0]*x + 1 <= 0;
cvx_end

disp("The optimal solution is attained at x* =");
disp(x);
disp("and the optimal value is: f(x*) = "+num2str(cvx_optval));

%% section b

Q = [5,2,0;2,4,1;0,1,7];
A = [1,0,0;0,1,0;0,0,0];
b = [0;0;1];
P = [1,0.5;0.5,1];
L = chol(P,'lower');
D = [zeros(2,1),L'];
cvx_begin
variable x(3)
minimize quad_form(x,Q) + norm([1,-1,0]*x,1)
subject to
    quad_over_lin([1,0,0]*x,[2,1,0]*x) + pow_pos(1+exp(norm(A*x+b,2)),7) - 200 <= 0;
    max([2,exp(pow_pos([1,1,0]*x,3))+quad_over_lin(D*x,[1,0,0]*x)+[-1,1,0]*x])+ [0,-2,0]*x <= 0;
    [-1,0,0]*x + 1 <= 0;
cvx_end

disp("The optimal solution is attained at x* =");
disp(x);
disp("and the optimal value is: f(x*) = "+num2str(cvx_optval));

