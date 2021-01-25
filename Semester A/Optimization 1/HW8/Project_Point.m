function [x_proj,u_proj] = Project_Point(x,n,m)
    x_proj = x(1:n);
    u_proj = zeros(n,m);
    for i = 1:m
        u = x((n*i+1):(n*(i+1)));
        if norm(u)<1
           u_proj(:,i) =  u;
        else
           u_proj(:,i) =  u/norm(u);
        end   
    end    
end
