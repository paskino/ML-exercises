% input:
all_theta = [1 -6 3; -2 4 -3];
X = [1 7; 4 5; 7 8; 1 4];
predictOneVsAll(all_theta, X)
fprintf("Expected values %output:");
fprintf(" 1\n  2\n  2\n  1\n");
