1;


function residual = linearChisq(thetas, X, Y)

  n = length (X);

  residual = 0.5/n * sum ( (linear_theta(thetas, X) - Y ).^2);

endfunction
function residual = linearChisqDiff(diff, X)

  n = length (X);

  residual = 0.5/n * sum ( ( diff ).^2);

endfunction


function residual = Chisq(func, X, Y)
  n = length (X);

  residual = 1/n * sum ( (func( X ) - Y ).^2);
endfunction

function ev = linear_theta(theta, x)
  ev = theta(1);
  for i=2:length(theta),
    ev = ev + x* theta(i);        
  end
endfunction

function result = regression_iter (alpha, thetas,  X, Y)
 n = length(X);
 diff = linear_theta(thetas, X) - Y;
 result = thetas(2:end) - alpha/n * sum( diff .* X);
 result = [thetas(1) - alpha/n * sum( diff ) , result];
endfunction

function result = regression_iter_diff (alpha, thetas, diff , X)
 n = length(X);
 %diff = linear_theta(thetas, X) - Y;
 result = thetas(2:end) - alpha/n * sum( diff .* X);
 result = [thetas(1) - alpha/n * sum( diff ) , result];
endfunction

%function f = example (thetas, X)
% f = thetas .* X';
%endfunction

%function ff = newfunc(thetas , f)
% ff = @f(thetas)
%endfunction

%linearChisq(thetas, X, Y)

% prepare a dataset
Xdata = 1:10:100;
noise = [1.18194
   1.07424
   0.69250
   1.16632
  -0.90812
  -1.57212
   1.92818
   0.39879
   0.96984
  -1.62558];

Y = (10- 2* Xdata) + 2.8*randn(1, length(Xdata));

% variable scaling to 0-1 range
theXdataRange = (max(Xdata) - min(Xdata));
X = (Xdata - min(Xdata)) / theXdataRange ;
%Y = (10- 2* X) + 0.8*noise'; 

% h_{theta} (x) = theta_0  + theta_1 * x 
% initial guess
thetas = [0.2,0.3]
j0 = linearChisq(thetas, X,Y);
diff = linear_theta(thetas, X) - Y;
iterations = 100000
residuals = j0
alpha = 0.0003
last_update = j0
adjust_alpha = 1

% stopping cryterion
epsilon = 1e-6
m = length(X);

tic;
for i=2:iterations
  last_diff = diff;
  tmp = regression_iter_diff(alpha, thetas, diff, X);
  %tmp = regression_iter(alpha, thetas, X, Y);
  thetas = tmp;
  diff = (linear_theta(thetas, X) - Y ) ;
  if sum( isnan(diff) ) > 1 
    i
    break
  end
  %j0 = linearChisq(thetas, X, Y);
  j0 = linearChisqDiff(diff, X);

  %if last_update - j0 < 0 && adjust_alpha == 1
  %  alpha = alpha * 3;
  %else
  %  alpha = alpha / 3;
  %  adjust_alpha = 0;
  %end
  
  residuals = [ residuals j0 ] ;
  if abs(last_diff - diff)/m < epsilon
   break;
  end
  %plot(residuals , '-*-' , ";residuals;")
end
toc;
subplot(2,1,1)
semilogy (residuals ,'-r' , 'linewidth' , 5)
title ('residuals')
xlabel('iteration')
ylabel('chisq')
subplot(2,1,2)
plot(X,Y,'or', 'markersize' , 10,X ,linear_theta(thetas,X),'linewidth' , 5)
