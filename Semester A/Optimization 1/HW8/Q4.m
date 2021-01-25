%% clear workspace
clc;
close all;
clear all;
%warning('off','all');

%% generate data
m = 5;                  % number of sensors
n = 2;                  % dimension of points

randn('seed', 317);
A = randn(n, m);        % matrix of sensors locations
x_real = randn(n,1);    % position of target
d = sqrt(sum((A - x_real*ones(1,m)).^2)) + 0.05*randn(1,m);
d = d';                 % distance measured from each sensor to target

%% define functions and gradiants

f = @(x,U) sum(vecnorm(x - A)'.^2 - 2 * d .* sum(U' .* (x - A)',2) + d.^2);
gradiant = @(x,U) 2 * [m*x-sum(A + U*diag(d),2);reshape(-(x-A)*diag(d),n*m,1)];


%% Projected Gradiant Constant step size

% find Lipshictz constant
B = m*eye(n);
C = zeros(n,n*m);
for i = 1:m
    C(:,((i-1)*n+1):(i*n)) = -d(i)*eye(n);
end
D = [B,C;C',zeros(m*n)];

% set Lipshictz constant
L = 2 * real(sqrt(max(eig(D'*D))));

% initial point
x0 = [1000;-500];
u0 = zeros(2,m);

% maximum allowed number of iterations
max_iterations = 100;

disp(" ");
disp("Projected_Gradiant_Constant:")

[x_k_constant,iterations,steps_constant,loss_constant] = Projected_Gradiant_Constant(f,gradiant,L,x0,u0,max_iterations);
disp("x0 = ("+num2str(x0(1))+","+num2str(x0(2))+") solution: (" ...
            +num2str(x_k_constant(1))+","+num2str(x_k_constant(2))+") iterations: "+num2str(iterations));


%% Projected Gradiant Backtracking
% backtracking parametrs
s = 1;
alpha = 1/2;
beta = 1/2;

% initial point
x0 = [1000;-500];
u0 = zeros(2,m);

% maximum allowed number of iterations
max_iterations = 100;

disp(" ");
disp("Projected_Gradiant_Backtracking:")

[x_k_backtracking,iterations,steps_backtracking,loss_backtracking] = Projected_Gradiant_Backtracking(f,gradiant,s,alpha,beta,x0,u0,max_iterations);
disp("x0 = ("+num2str(x0(1))+","+num2str(x0(2))+") solution: (" ...
            +num2str(x_k_backtracking(1))+","+num2str(x_k_backtracking(2))+") iterations: "+num2str(iterations));

%% plot results

figure();
p1 = semilogy(0:max_iterations,vecnorm(x_real - steps_constant),'r');
hold on;
p2 = semilogy(0:max_iterations,loss_constant,'--r');
p3 = semilogy(0:max_iterations,vecnorm(x_real - steps_backtracking),'b');
p4 = semilogy(0:max_iterations,loss_backtracking,'--b');
grid on;
title("Error vs. Iterations");
xlabel("Iteration Number");
ylabel("Value");
legend([p1 p2 p3 p4],{'Relative error constant','Objective Function constant','Relative error Backtracking','Objective Function Backtracking'});


figure();
% plot sensors locations
p1 = scatter(A(1,:),A(2,:),'r','filled');
hold on;

% plot constant stepsize result
p2 = scatter(x_k_constant(1),x_k_constant(2),'g','filled');

% plot backtracking result
p3 = scatter(x_k_backtracking(1),x_k_backtracking(2),'b','filled');

% plot backtracking result
p4 = scatter(x_real(1),x_real(2),'m','filled');
grid on;
xlabel("X coordinate");
ylabel("Y coordinate");
title("Sensors,real, and solutions locations");
legend([p1 p2 p3 p4],{'sensors locations','constant step size','backtracking','real location'});
