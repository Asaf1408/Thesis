function [x_k,iterations,converged] = Gradiant_Backtracking(func,grad_func,s,alpha,beta,x0)
    
    converged = 1;
    
    % maximum allowed number of iterations
    max_iterations = 10000;
    
    % set initial point
    x_k = x0;
    
    % run until stopping criteria is satisfied
    iterations = 0;
    while ((norm(grad_func(x_k))>1e-5) && (iterations < max_iterations))
        iterations = iterations + 1;
        %disp("iteration: "+num2str(iterations));
        % set initial backtracking step size    
        t_k = s;
        
        % find sufficient backtracking step size
        while (func(x_k)-func(x_k-t_k*grad_func(x_k)) < alpha * t_k * norm(grad_func(x_k))^2)
            t_k = beta * t_k;
        end
        
        % update solution
        x_k = x_k - t_k * grad_func(x_k);
    end
    
    % check if convergence achived
    if (iterations == max_iterations)
        converged = -1;
    end    
end

