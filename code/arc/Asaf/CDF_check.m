%% clean working enviorment
close all;
clear all;
clc;

%% plot graphs
axis = 0.8:0.001:1;
ratios = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1];
figure();
for ratio = ratios
    y = normcdf(norminv(axis)+ratio);  
    plot(axis,y);
    hold on;
end  
grid on;
grid minor;
xlabel("Quantile");
ylabel("New Quantile");
title("\Phi(\Phi^-^1(Q)+ratio)");
legend(string(ratios));
