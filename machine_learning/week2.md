# (Multivariate Linear Regression) Linear Regression with Multiple Variables
* Variables represent the features used in a prediction.

#### New Hypothesis Function
* h(x) = theta<sub>0</sub> + theta<sub>1</sub> * x<sub>1</sub> + ... + theta<sub>n</sub> * x<sub>n</sub>
  * x = vector[x<sub>0</sub>, x<sub>1</sub>, ... x<sub>n</sub>] for real numbers from 0 to N + 1
  * theta = vector[theta<sub>0</sub>, theta<sub>1</sub>, ... theta<sub>n</sub>] for real numbers from 0 to N + 1
    * Think of theta parameters as a N + 1 dimensional vector.
    * Cost function is now J(theta) instead of J(theta<sub>0</sub>, theta<sub>1</sub>, ... theta<sub>n</sub>)
* h(x) = theta<sup>T</sup> * x

* Gradient Descent for cost function is generalized to:
  * theta<sub>feature</sub> = theta<sub>feature</sub> - learning\_rate
    * partial\_deriv. w.r.t. theta<sub>feature</sub> of J(theta)

#### Optimizing Gradient Descent with Feature Scaling & Mean Normalization
* Make sure features are on a similar scale so gradient descent can converge
  quicker.
  * Example: feature<sub>1</sub>: size (0 ~ 2000 ft.<sup>2</sup>)
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

#### Determing Value of Learning Rate
* Making sure gradient descent works correctly:
  * lim (J(theta)) as theta -> inf. = some converged value s.t. J(theta) is
    lower than any J(theta') for fewer iterations. 
  * Lemma: J(theta) should decrease at **every** iteration.
* Common cause of diverging J(theta): large values for learning rate.
* Good measurement for learning rate: three-fold increases (0.001, 0.003, 0.01,
  0.03, 0.1, 0.3, 1, ...)
* Good measurement for testing convergence point reached: J(theta) decreases by
  less than 10<sup>-3</sup> in a single iteration.

#### Features and Polynomial Regression
* Polynomial Regression: creation of new features based on existing features, either to optimize J(theta) or fit non-linear functions (linear regression doesn't require a straight line for an optimized model, since it''s all dependent on the training set).
  * I.e. calculating Area feature based on frontage & depth (frontage * depth)
    * hypothesis = theta<sub>0</sub> + theta<sub>1</sub> * x<sub>1</sub> where
      x<sub>1</sub> is the "Area" feature.
  * I.e. given a quadratic relationship between size and price of a house, use
    the hypothesis theta<sub>0</sub> + theta<sub>1</sub> * x + theta<sub>2</sub> * x<sup>2</sup> 
    * Feature(s) are simply exponentiated up to degree N to model the new
      hypothesis function.
      * I.e. if x = size, the term theta<sub>2</sub> * x<sup>2</sup> is
        theta<sub>2</sub> * (size)<sup>2</sup>
      * **NOTE**: Feature scaling will suddenly become very important through
        use of polynomial regression (make sure to get features to comparable
values).
        * Ex.: size of range(0, 1000) with a new hypothesis function h(theta)
          = theta<sub>0</sub> + theta<sub>1</sub> * size + theta<sub>2</sub>
* sqrt(size) feature scales size to be size / 1000 for theta<sub>1</sub> and sqrt(size) / 32 for theta<sub>2</sub> (sqrt(1000) ~ 32).
* Takeaway: If model can't be as predictive with given features, can choose
  different features **using** existing features to *potentially* get better
models. 
* Will later learn algorithms to determine best transformations for feature to
  optimize model (if existing features are not optimal).

