Theta1 = reshape(sin(0 : 0.5 : 5.9), 4, 3);
Theta2 = reshape(sin(0 : 0.3 : 5.9), 4, 5);
X = reshape(sin(1:16), 8, 2);
p = predict(Theta1, Theta2, X)
fprintf(" you should see this result\n")
fprintf("p =\n") 
i = 1;
fprintf("  4 %d\n", p(i++));
fprintf("  1 %d\n", p(i++));
fprintf("  1 %d\n", p(i++));
fprintf("  4 %d\n", p(i++));
fprintf("  4 %d\n", p(i++));
fprintf("  4 %d\n", p(i++));
fprintf("  4 %d\n", p(i++));
fprintf("  2 %d\n" , p(i++));
