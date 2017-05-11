# (Multivariate Linear Regression) Linear Regression with Multiple Variables
* Variables represent the features used in a prediction.

#### New Hypothesis Function
* h(x) = theta0 + theta1 * x1 + ... + thetaN * xN
  * x = vector[x0, x1, ... xN] for real numbers from 0 to N + 1
  * theta = vector[theta0, theta1, ... thetaN] for real numbers from 0 to N + 1
    * Think of theta parameters as a N + 1 dimensional vector.
    * Cost function is now J(theta) instead of J(theta0, theta1, ... thetaN)
* h(x) = theta<sup>T</sup> * x

* Gradient Descent for cost function is generalized to:
  * theta<sub>feature</sub> = theta<sub>feature</sub> - learning\_rate
    * partial\_deriv. w.r.t. theta<sub>feature</sub> of J(theta)

#### Optimizing Gradient Descent with Feature Scaling & Mean Normalization
* Make sure features are on a similar scale so gradient descent can converge
  quicker.
  * Example: feature<sub>1</sub>: size (0 ~ 2000 ft.^2)
             feature<sub>2</sub>: number of bedrooms (1 - 5)
    * Result: gradient descent ends up oscillating to find global minimum.
    * Solution: divide by range (i.e. size / (2000 - 0))
      * 0 <= feature <= 1
    * Goal: Get every feature into approx. -1 <= feature <= 1

* Mean Normalization: replace feature<sub>i</sub> with feature<sub>i</sub>
  - mew<sub>i</sub> to make features have approximately zero mean. 
  * mew<sub>i</sub>: average value of feature<sub>i</sub> in the training set.

* Takeaway: small ranges for theta vector causes a quicker descent, whereas
  a large range descends slowly due to heavy oscillation. Restricting to
a semi-fixed range (-1 <= feature<sub>i</sub> <= 1) allows for theta to take
precise (but not slow) descents per iteration.

