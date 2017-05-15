% Julian Domingo : jad5348

function A_inv_b = matrixInverseVector(A, b, x_init, alpha)
  % Uses gradient descent to avoid calculating the inverse matrix to find A^-1
  % * b, error free up to 10 .^ -6 decimal places.
  x = x_init;
  while ((norm((A * x) - b) .^ 2) < (10 .^ -6)),
    x = x - alpha * (2 * A * ((A * x) - b)); % term multiplied by alpha is the gradient of f(x)
  end; 
  A_inv_b = x;
endfunction
