% Julian Domingo : jad5348

function cost = SquaredErrorCost(A, b, x)
  cost = norm((A * x) - b) .^ 2;
endfunction
