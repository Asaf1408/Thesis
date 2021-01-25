function [x_k,iterations,steps,loss_func] = Projected_Gradiant_Backtracking(func,grad_func,s,alpha,beta,x0,u0,max_iterations)
                
    % get dimension and number of sensors
    n = numel(x0);
    m = numel(u0(1,:));
    
    % set initial point
    x_k = x0;
    u_k = u0;
    
    % set containers for realtive error and loss function
    steps = zeros(2,max_iterations+1);
    steps(:,1) = x0;
    loss_func = zeros(max_iterations+1,1);
    loss_func(1) = func(x_k,u_k);
    
    % run until stopping criteria is satisfied
    iterations = 0;
    while (iterations < max_iterations)
        iterations = iterations + 1;
        % set initial backtracking step size    
        t_k = s;
        
        % find sufficient backtracking step size
        while (true)
            [new_x,new_u] = Project_Point([x_k;u_k(:)]-t_k*grad_func(x_k,u_k),n,m);
            if (func(x_k,u_k)-func(new_x,new_u)) <  (alpha * t_k * norm((1/t_k)*([x_k;u_k(:)]-[new_x;new_u(:)])).^2)
                t_k = beta * t_k;
            else
                break;
            end    
            
        end
        
        % update solution
        x_k = new_x;
        u_k = new_u;
        steps(:,iterations+1) = x_k;
        loss_func(iterations+1) = func(x_k,u_k);
    end
      
end
