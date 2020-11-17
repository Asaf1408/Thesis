%% clear workspace
clc;
close all;
clear all;

%% define functions and gradiants
f1 = @(x) -13 + x(1) + ((5-x(2))*x(2)-2)*x(2);
f1_x = @(x) 1;
f1_y = @(x) -3*(x(2).^2) + 10*x(2) - 2;
f1_yy = @(x) -6*x(2) + 10;

f2 = @(x) -29 + x(1) + ((x(2)+1)*x(2)-14)*x(2);
f2_x = @(x) 1;
f2_y = @(x)  3*(x(2).^2) + 2*x(2) - 14;
f2_yy = @(x) 6*x(2) + 2;

f = @(x) f1(x)^2 + f2(x)^2;
f_x = @(x) 2*(f1(x) + f2(x));
f_y = @(x) 2*(f1(x)*f1_y(x) + f2(x)*f2_y(x));
f_xx = @(x) 2*(f1_x(x) + f2_x(x));
f_xy = @(x) 2*(f1_y(x) + f2_y(x));
f_yy = @(x) 2*(f1_y(x)^2 + f1(x)*f1_yy(x) + f2_y(x)^2 + f2(x)*f2_yy(x));
gradiant = @(x) [f_x(x);f_y(x)];
hessian = @(x) [f_xx(x),f_xy(x);f_xy(x),f_yy(x)];
Jacobian = @(x) [f1_x(x),f1_y(x);f2_x(x),f2_y(x)];


%% Gradiant_Backtracking
s = 1;
alpha = 1/2;
beta = 1/2;
initial_points = [-50,20,20,5;7,7,-18,-10];
Num_of_initial_points = numel(initial_points(1,:));

disp(" ");
disp("Gradiant_Backtracking:")
for j = 1:Num_of_initial_points
    x0 = initial_points(:,j);
    [x_k,iterations,converged] = Gradiant_Backtracking(f,gradiant,s,alpha,beta,x0);
    if (converged == 1)
        disp("x0 = ("+num2str(x0(1))+","+num2str(x0(2))+") solution: (" ...
            +num2str(x_k(1))+","+num2str(x_k(2))+") iterations: "+num2str(iterations));
    else
        disp("x0 = ("+num2str(x0(1))+","+num2str(x0(2))+") not converged");
    end    
end

%% Gradiant_Newton_Backtracking
s = 1;
alpha = 1/2;
beta = 1/2;
initial_points = [-50,20,20,5;7,7,-18,-10];
Num_of_initial_points = numel(initial_points(1,:));

disp(" ");
disp("Gradiant_Newton_Backtracking:")
for j = 1:Num_of_initial_points
    x0 = initial_points(:,j);
    [x_k,iterations,converged] = Gradiant_Newton_Backtracking(f,gradiant,hessian,s,alpha,beta,x0);
    if (converged == 1)
        disp("x0 = ("+num2str(x0(1))+","+num2str(x0(2))+") solution: (" ...
            +num2str(x_k(1))+","+num2str(x_k(2))+") iterations: "+num2str(iterations));
    else
        disp("x0 = ("+num2str(x0(1))+","+num2str(x0(2))+") not converged");
    end    
end

%% Guass_Newton_Backtracking
s = 1;
alpha = 1/2;
beta = 1/2;
initial_points = [-50,20,20,5;7,7,-18,-10];
Num_of_initial_points = numel(initial_points(1,:));

disp(" ");
disp("Guass_Newton_Backtracking:")
for j = 1:Num_of_initial_points
    x0 = initial_points(:,j);
    [x_k,iterations,converged] = Guass_Newton_Backtracking(f,gradiant,Jacobian,s,alpha,beta,x0);
    if (converged == 1)
        disp("x0 = ("+num2str(x0(1))+","+num2str(x0(2))+") solution: (" ...
            +num2str(x_k(1))+","+num2str(x_k(2))+") iterations: "+num2str(iterations));
    else
        disp("x0 = ("+num2str(x0(1))+","+num2str(x0(2))+") not converged");
    end    
end

