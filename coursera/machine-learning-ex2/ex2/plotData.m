function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%



% a = [1; 0; 1]
% b = [1; 2; 3]
% c = b(any(a,2))
plot (X(any(y,2),1),X(any(y,2),2), 'g+')
plot (X(any((1-y),2),1),X(any((1-y),2),2), 'ro')






% =========================================================================



hold off;

end
