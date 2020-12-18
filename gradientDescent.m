function [theta, J_history] = gradientDescent(X, y, theta, alpha,iterations)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
n=size(X);
J_history = zeros(iterations, 1);
for iter = 1:iterations
     sum1=0;
     sum2=0;
    for i=1:m
        sum1=sum1+(theta(1)+theta(2)*X(i,2)-y(i));
        sum2=sum2+(theta(1)+theta(2)*X(i,2)-y(i))*X(i,2);
    end
    sum1=sum1/m;
    sum2=sum2/m;
    theta(1)=theta(1)-alpha*sum1;
    theta(2)=theta(2)-alpha*sum2;
    J_history(iter) = computeCost(X, y, theta);
   

end
end
