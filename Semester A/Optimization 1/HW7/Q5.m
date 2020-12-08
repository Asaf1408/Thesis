%% cleaning the worksapce
clear all;
close all;
clc;

%% generate data
m = 50;
n = 2;
outliers_num = 10;
rand('seed',314);
A = 3000*rand(n,m);
A(:,1:outliers_num) = A(:,1:outliers_num)+3000;
p = round(10*rand(m,1)+10)';
alpha = 0.01;
gamma = 1.2;
eta1 = 20;
eta2 = 30;
mu1 = 2;
mu2 = 5;

%% section a

cvx_begin
variable x(n)
minimize alpha * gamma * sum(p .* norms(repmat(x,1,m)-A,2,1))

cvx_end

disp("The optimal solution is attained at x* =");
disp(x);
disp("and the optimal value is: f(x*) = "+num2str(cvx_optval));

x_val = cvx_optval;


%% section b

cvx_begin
variable y(n)
minimize sum(gamma * alpha * p .* norms(repmat(y,1,m)-A,2,1) +...
             mu1 * p .* max(zeros(1,m),alpha*norms(repmat(y,1,m)-A,2,1)-eta1)+...
             (mu2-mu1) *p .* max(zeros(1,m),alpha*norms(repmat(y,1,m)-A,2,1)-eta2))

cvx_end

disp("The optimal solution is attained at y* =");
disp(y);
disp("and the optimal value is: f(y*) = "+num2str(cvx_optval));

y_val = cvx_optval;


%% section c

c1 = (1/m) * sum(A,2)';
c2 = (1/m) * sum(vecnorm(A).^2);

cvx_begin
variable z(n)
minimize max(abs(2 * (repmat(c1,m,1) - transpose(A)) * z +transpose((vecnorm(A).^2)) - repmat(c2,m,1)))

cvx_end

disp("The optimal solution is attained at z* =");
disp(z);
disp("and the optimal value is: f(z*) = "+num2str(cvx_optval));

z_val = cvx_optval;

%% plot results

figure();
p1 = scatter(A(1,:),A(2,:),'r');
hold on;
p2 = scatter(x(1),x(2),'b','filled');
p3 = scatter(y(1),y(2),'g','filled');
p4 = scatter(z(1),z(2),'m','filled');
grid on;
xlabel("X coordinate");
ylabel("Y coordiante");
legend([p1,p2,p3,p4],"demend points","solution 1","solution 2","solution 3");
title("Optimal warehouse locations");

text(4000,2000,"sol 1 func val: "+num2str(x_val)+newline+newline+"sol 2 func val: "+num2str(y_val)+...
                newline+newline+"sol 3 func val: "+num2str(z_val));


