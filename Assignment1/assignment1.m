%% Question 1

%plotting implicit function -> pnorm1(v1,v2,p)-1 = 0
fimplicit(@(v1,v2) pnorm1(v1,v2,0.5)-1,[-2 2 -2 2],'r');
hold on
fimplicit(@(v1,v2) pnorm1(v1,v2,2)-1,'g');
fimplicit(@(v1,v2) pnorm1(v1,v2,4)-1,'b');
fimplicit(@(v1,v2) pnorm1(v1,v2,1000)-1,'m');
fimplicit(@(v1,v2) pnorm1(v1,v2,1)-1,'r');

hold off

%% Question 2

%initializing
y1 = rand(10, 1);
x1 = rand(10, 1);

%3D Plot
fimplicit3(@(m,c,e) sum((y1 - (m*x1 + c)).^2) - e);
title('3D Graph - Error vs Slope m and Intercept c');
xlabel('Slope (m)');
ylabel('Intercept (c)');
zlabel('Error (e)');

%2D PLot
figure
grid on
fimplicit(@(m,e) sum((y1 - (m*x1)).^2) - e);
title('2D Graph - Error vs Slope m');
xlabel('Slope (m)');
ylabel('Error (e)');


%% Question 3

%initializing values
m = length(y1);
theta = zeros(2,1);
iterations = 1000;
alpha = 0.01;

%plotting y1 with x1
figure
plot(x1,y1,'o','MarkerSize',5);
title('2D Graph - Error vs Slope m');
xlabel('x1');
ylabel('y1');

% Compute the Cost Function
X = [ones(m, 1), x1];
J = costfunction(X, y1, theta);

% Run Gradient Descent
[theta, Js] = gradient_descent(X, y1, theta, alpha, iterations);
hold on;
plot(X(:, 2), X * theta, '-');
legend('Training data', 'Linear regression');
hold off;


