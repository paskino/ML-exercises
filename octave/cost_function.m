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
 m = length(X);
 diff = linear_theta(thetas, X) - Y;
 result = thetas(2:end) - alpha/m * (diff' * X);
 result = [thetas(1) - alpha/m * sum( diff ) , result];
endfunction

%update rule for the fit
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
Xdata = -3:.5:5;
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
current_diff = linear_theta(thetas, X) - Y;
iterations = 1000
residuals = j0
alpha = 1e+0
adjust_alpha = [alpha]

% stopping cryterion
epsilon = 1e-7
m = length(X);

tic;
for i=2:iterations
  last_diff = current_diff;
  tmp = regression_iter_diff(alpha, thetas, current_diff, X);
  %tmp = regression_iter(alpha, thetas, X, Y);
  thetas = tmp;
  current_diff = (linear_theta(thetas, X) - Y ) ;
  if sum( isnan(current_diff) ) > 1 
    i
    break
  end
  %j0 = linearChisq(thetas, X, Y);
  j0 = linearChisqDiff(current_diff, X);
  
  down_step = j0 - residuals(i-1);
  if down_step < 0
    
    if (down_step ) > - m * 1e-0
      alpha = alpha * 3;
    end
  else
    alpha = alpha / 3;
  end
  adjust_alpha = [adjust_alpha alpha];
  residuals = [ residuals j0 ] ;
  if abs(down_step) < epsilon
   break;
  end
  %plot(residuals , '-*-' , ";residuals;")
end
toc;
subplot(2,3,1)
semilogy (residuals ,'-r' , 'linewidth' , 5 )  
subplot(2,3,4)
plot (diff(residuals) ,'-r' , 'linewidth' , 5 )  
title ('residuals')
ylabel('chisq')
subplot(2,3,[2 5])
semilogy (adjust_alpha,'.g' , 'linewidth' , 5 )
ylabel('alpha')
xlabel('iteration')
subplot(2,3,[3 6])
plot(X,Y,'.r', 'markersize' , 4,X ,linear_theta(thetas,X),'linewidth' , 5)

% the results of the linear fit are
thetas(1) - (min(Xdata)* thetas(2) / theXdataRange)
thetas(2)/theXdataRange
