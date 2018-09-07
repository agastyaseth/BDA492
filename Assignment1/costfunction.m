function [ J ] = costfunction( X,y,theta )

    m = length(y);
    %hypothesis vector - h
    h = X * theta;
    
    %cost function
    J = 1/(2*m) * sum((h-y).^2);

end