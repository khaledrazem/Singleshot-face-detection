function Y = Cleaneps(Y)

% Get precision of prediction to prevent errors due to floating point
% precision.
precision = underlyingType(Y);

% Convert values less than floating point precision to eps.
Y(Y < eps(precision)) = eps(precision);

% Convert values between 1-eps and 1 to 1-eps.
Y(Y > 1 - eps(precision)) = 1 - eps(precision);


end