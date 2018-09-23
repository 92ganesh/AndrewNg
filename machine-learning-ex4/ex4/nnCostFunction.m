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
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Part 1
X = [ones(m, 1) X];
layer2 = sigmoid(X * Theta1');
m = size(layer2, 1);
layer2 = [ones(m, 1) layer2];
outputLayer = sigmoid(layer2*Theta2');

c = size(outputLayer);

oneHot = zeros(1,c(2));
oneHot(1,y(1,1)) = 1;

for i=2:m
    oneHot2 = zeros(1,c(2));
    oneHot2(1,y(i,1)) = 1;
    oneHot=[oneHot;oneHot2];
end

error = -(oneHot.*log(outputLayer)+(1.-oneHot).*log(1.-outputLayer));
final  = sum(error) ;
J  = sum(final)/m ;

% regularization
%leaving the bias term
last1=size(Theta1);
last1=last1(2);
regTheta1 = Theta1(:,2:last1);
regTheta2 = Theta2(:,2:end);
regularization = sum(sum(regTheta1.^2))+sum(sum(regTheta2.^2));
regularization = (lambda*regularization)/(2*m);
J = J + regularization;

% Part 2
delta3 = outputLayer-oneHot;
delta3=delta3';
fprintf('-----\n');
size(Theta2')
size(delta3)
fprintf('-----\n');
delta2 = Theta2'*delta3.*sigmoidGradient(layer2);

delta2=delta(2:end);
Theta2_grad=(delta3*layer2')/m;
Theta1_grad=(delta2*X')/m;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
