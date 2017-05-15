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

* **NOTE** : range is min and max of **values of the feature in training set only**, so for a grade on an exam with two samples with grades 40 and 100, the range is **NOT** (100 - 0) and instead (100 - 40).

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

#### Computing Parameters Analytically
* Normal equation: gives us better way to solve for theta parameters for some
  linear regression problems (applicable only for linear regression problems)
  * Solve for theta analytically (don't need to do it iteratively like with
    gradient descent).
* Intuitively, if theta was just 1D (constant, not a vector), then can solve
  for theta by taking derivate of J(theta).
* For theta of higher dimensions:
  * theta = (X<sup>T</sup>X)<sup>-1</sup>X<sup>T</sup>y; where y = output
    vector and X = design matrix (rows are feature vectors of each sample, transposed).
  * Ex.: 1 feature with 1 sample => design matrix = [1, x<sup>1</sup>], y = [result<sub>sample1</sub>]
* Feature scaling isn't necessary for calculating theta using normal equation.
* Choosing Gradient Descent vs. Normal Equation:
  * Gradient Descent: 
    * -: need to choose learning rate, needs many iterations
    * +: scalable (works well even when many features) O(kn<sup>2</sup>); k = number of iterations
  * Normal equation:
    * -: Need to compute X<sup>T</sup>X<sup>-1</sup>, and cost of inverting matrices is O(feature\_size<sup>3</sup>)
    * +: don't need to choose learning rate, don't need to iterate 
  * Threshold point: ~10<sup>3</sup> features > for gradient descent

#### Non-invertibility of Normal Equations
* What happens if design matrix is non-invertible / degenerate? 
  * Octave: pinv() handles degenerate matrices correctly.
* Common causes of degenerate matrices:
  * Redundant features (linearly dependent): size in different metrics
  * Too many features s.t. size(training set) < size(features): delete some or use regularization (see later weekX.md)
 
#### Using Octave
* Matrices: M = [1 2; 3 4; 5 6] == 3 x 2 matrix
  * All ones matrix: ones(row, column)
  * All zeros matrix: zeros(row, column)
  * ";" is column delimiter
  * Identity matrix: eye(dimension)
* Vector: V = [1 2 3] == 1 x 3 vector
  * Shorthand column vectors: 1:.5:2 == [1 1.5 2]
    * Partition defaults to 1 
* randn(3 , 3) = 3 x 3 matrix of random numbers drawn from normal / gaussian distribution
  * rand(row, col) is for uniform distribution
* Plotting:
  * hist(vector, <optional bin count>) 

#### Loading Data in Octave
* size(matrix) => <row dimension> <column dimension>
  * size(matrix, 1=row size or 2=col size)
* length(vector) => column size of vector
* pwd
* load('file name') 
  * "file name" => prints out data
  * "file name" will be a matrix/vector
* who/whos: variables in workspace
* clear "variable": deletes variable from memory
  * no variable param == deletes all variables
* Saving variables to disk (binary format):
  * save "file name" "variable"
  * Readable format (ASCII): save "file name" v -ascii
* Indexing matrices:
  * M(row, col): **1-indexed**
  * M(2,:): similar to python -> get entire row
  * M(:, 2): get entire column
  * M([1 2 3], :): get entire rows 1, 2, 3
* Replacement in matrices:
  * M(:,2) = [10; 11; 12]: replaces second column with column vector [10, 11,
    12]
* Appending in matrices:
  * M = M[M, [1; 2; 3]]: appends matrix M with a new column [1, 2, 3] 
* Flatten matrices (puts into a single column vector):
  * M(:)
* Concatenating matrices:
  * C = [A B] where A and B are matrices of same dimensions
  * C = [A; B]: 
**semicolon means put in the bottom, comma means append / put to the right**


