function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_vec = [0.1 0.3 1 3 10 20 30]';
sigma_vec = [0.03 0.1 0.3 1 3]';

error_train = zeros(length(C_vec), length(sigma_vec));
error_val = zeros(length(C_vec), length(sigma_vec));
min_error_train = 0;
min_error_val = 0;

for i = 1:length(C_vec)
 for j = 1:length(sigma_vec)
  C = C_vec(i)
  sigma = sigma_vec(j)
  %theta = trainLinearReg([ones(size(X,1),1) X], y, lambda);
  model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
  predictions = svmPredict(model, Xval);
  error_val(i,j) = mean(double(predictions ~= yval));
  predictions = svmPredict(model, X);
  error_train(i,j) = mean(double(predictions ~= y));
  %
  if i == 1 && j ==1
    b_C = C;
    b_sigma = sigma;
    min_error_val = error_val(i,j);
  endif
  
  if min_error_val > error_val(i,j)
    b_C = C
    b_sigma = sigma
    min_error_val = error_val(i,j)
  endif
  
  
 endfor
endfor

sigma = b_sigma;
C = b_C;

% =========================================================================

end
