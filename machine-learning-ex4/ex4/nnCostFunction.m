function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
%{ 1. Feed-forward to compute h = a3.
a1 = [ones(1, m); X'];  % 401 x m
z2 = Theta1 * a1;
a2 = [ones(1, m); sigmoid(z2)];  % 26 x m
a3 = sigmoid(Theta2 * a2);  % 10 x m
h = a3;

Y = zeros(10,size (y));
for i= 1:m
Y(y(i,1),i) = 1;
end
J = J + (lambda / (2*m)) * sum(sum(Theta1(:, 2:end) .^ 2));
J = J + (lambda / (2*m)) * sum(sum(Theta2(:, 2:end) .^ 2));

% these works above

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%{
for i=1:m
  x = X (i,:);
  yback = Y(:,i);   
  yback = yback';   %1x10 matrix
  a1 = x;
  a1 = [ones(1,size(a1,1));(a1)']';  % adding biased parameter
  z2 = a1*Theta1';
  a2 = sigmoid(z2);
  a2 = [ones(1,size(a2,1));(a2)']';  % adding biased parameter
  z3 = a2*Theta2';    
  a3 = sigmoid(z3);               %1x10 matrix
  d3 = (a3 - yback);
  z2 = [ones(1,size(z2,1));(z2)']';
  d2  = (d3*Theta2).*sigmoidGradient(z2);                %1x26
   Theta2_grad = (Theta2_grad + d3' * a2);
   A = d2'*a1;
   Theta1_grad = (Theta1_grad + A(2,:));
   
 % D2 = (1/m)*(delta2+lambda*Theta2);
  %D1 = (1/m)*(delta1);
 endfor;
 }

d3 = a3 - Y;  % 10 x m
d2 = (Theta2' * d3) .* [ones(1, m); sigmoidGradient(z2)];  % 26 x m

% Vectorized ftw:
Theta2_grad = (1/m) * d3 * a2';
Theta1_grad = (1/m) * d2(2:end, :) * a1';

% Add gradient regularization.
Theta2_grad = Theta2_grad + ...
              (lambda / m) * ([zeros(size(Theta2, 1), 1), Theta2(:, 2:end)]);
Theta1_grad = Theta1_grad + ...
              (lambda / m) * ([zeros(size(Theta1, 1), 1), Theta1(:, 2:end)]);




% Part 3: Implement regularization with the cost function and gradients.
%
%        Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%{
d2 = d2';
Theta2_grad = (1/m) * d3' * a2;
Theta1_grad = (1/m) * d2(2:end, :) * a1;

% Add gradient regularization.
Theta2_grad = Theta2_grad + ...
              (lambda / m) * ([zeros(size(Theta2, 1), 1), Theta2(:, 2:end)]);
Theta1_grad = Theta1_grad + ...
              (lambda / m) * ([zeros(size(Theta1, 1), 1), Theta1(:, 2:end)]);


%}
%}
%}
a1 = [ones(1, m); X'];  % 401 x m
z2 = Theta1 * a1;
a2 = [ones(1, m); sigmoid(z2)];  % 26 x m
a3 = sigmoid(Theta2 * a2);  % 10 x m

% Explode y into 10 values with Y[i] := i == y.
Y = zeros(num_labels, m);
Y(sub2ind(size(Y), y', 1:m)) = 1;

% Compute the non-regularized error. Fully vectorized, at the expense of
% having an expanded Y in memory (which is 1/40th the size of X, so it should be
% fine).
J = (1/m) * sum(sum(-Y .* log(a3) - (1 - Y) .* log(1 - a3)));

% Add regularized error. Drop the bias terms in the 1st columns.
J = J + (lambda / (2*m)) * sum(sum(Theta1(:, 2:end) .^ 2));
J = J + (lambda / (2*m)) * sum(sum(Theta2(:, 2:end) .^ 2));


% 2. Backpropagate to get gradient information.
d3 = a3 - Y;  % 10 x m
d2 = (Theta2' * d3) .* [ones(1, m); sigmoidGradient(z2)];  % 26 x m

% Vectorized ftw:
Theta2_grad = (1/m) * d3 * a2';
Theta1_grad = (1/m) * d2(2:end, :) * a1';

% Add gradient regularization.
Theta2_grad = Theta2_grad + ...
              (lambda / m) * ([zeros(size(Theta2, 1), 1), Theta2(:, 2:end)]);
Theta1_grad = Theta1_grad + ...
              (lambda / m) * ([zeros(size(Theta1, 1), 1), Theta1(:, 2:end)]);












%}

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
