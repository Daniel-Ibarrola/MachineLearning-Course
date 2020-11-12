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

%Values for C and Sigma
values = [0.01 0.03 0.1 0.3 1 3 10 30];
%Initialize error matrix and a counter variable
error_matrix = zeros(64,3);
count = 0;
%Iterate over each combination of the parameters
for n=1 : length(values)
    for i=1 : length(values)
        
        C_val = values(n);
        sigma_val = values(i);
        
        model = svmTrain(X, y, C_val, @(x1, x2)gaussianKernel(x1, x2, sigma_val));
        predictions = svmPredict(model, Xval);
        pred_error =  mean(double(predictions ~= yval));
        
        count = count + 1;
        
        error_matrix(count,1) = pred_error;
        error_matrix(count,2) = n;
        error_matrix(count,3) = i;
    end
end

[mval idx] = min(error_matrix(:,1));

iC  = error_matrix(idx, 2);
isigma = error_matrix(idx, 3);

C = values(iC);
sigma = values(isigma);
% =========================================================================

end
