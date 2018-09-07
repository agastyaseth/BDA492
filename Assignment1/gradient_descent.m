function [ theta,Js ] = gradient_descent( X,y,theta,alpha,iterations )
    m = length(y);
    Js = zeros(iterations, 1);
    
    for i = 1 : iterations
        h = X * theta;
        t1 = theta(1) - (alpha * (1 / m) * sum(h - y));
        t2 = theta(2) - (alpha * (1 / m) * sum((h - y) .* X(:, 2)));
        theta(1) = t1;
        theta(2) = t2;
        
        Js(i) = costfunction(X, y, theta);
    end
end