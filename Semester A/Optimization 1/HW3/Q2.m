%% clean work space
clear all
close all
clc

%% section b 

% get data points
X = curve()';

% estimate coeeficents using least squares
[c0,c1,c2,d1,d2] = fit_rational(X);

% plot curve and data points
x = -5:0.01:5;
y1 = (c0+c1.*x+c2.*(x.^2))./(1+d1.*x+d2.*(x.^2));
figure();
plot(x,y1,'r');
hold on;
scatter(X(:,1),X(:,2),'xb');
grid on;
xlabel('x');
ylabel('y');
title('estiamted curve');
legend("estimated curve","data points");

%% section d 

% estimate coeeficents using Rigly quetiont
[c0,c1,c2,d0,d1,d2] = fit_rational_normed(X);

% plot curve
y2 = (c0+c1.*x+c2.*(x.^2))./(d0+d1.*x+d2.*(x.^2));
plot(x,y2,'g');

title('estiamted curve');
legend("Least Squares","data points","Rayleigh Quotient");