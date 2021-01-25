function [x_k,iterations,steps,loss_func] = Projected_Gradiant_Constant(func,grad_func,L,x0,u0,max_iterations)
                 
    % get dimension and number of sensors
    n = numel(x0);
    m = numel(u0(1,:));
    
    % set constant step size according to Lipshitz constant
    t = 1/L;
    
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
              
        % update solution
        [x_k,u_k] = Project_Point([x_k;u_k(:)]-t*grad_func(x_k,u_k),n,m);
        steps(:,iterations+1) = x_k;
        loss_func(iterations+1) = func(x_k,u_k);
    end
      
end

