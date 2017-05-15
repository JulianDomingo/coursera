# ML Algorithm Classifications
* Supervised
* Unsupervised
* Recommender Systems    

### Supervised Learning

__Example__: Housing price prediction. 
* Choosing straight line or quadratic to data set to accurately determine house price.
* __Regression Model__: map input variables to some continuous function. 

__Example 1__: Breast Cancer (malignant or benign?)
* Data set contains benign (vertical value == 0) and malignant (== 1) points.
* __Classification Model__: map input variables into discrete categories (discrete valued output (Map values to classifications)).

* Regression algorithms: predict continuous valued output, "right answers"
  given for each sample in the data set.
* Classification algorithms: discrete valued output (i.e. benign or malignant
  tumors?)

### Unsupervised Learning
* Data has no labels, not told what to do with it. 
* Goal: find structure in this data.
* There is no "right answer" in the same sense as supervised learning. 
* Derive structure by clustering the data based on relationships among the
  variables in the data.

## Types of Unsupervised Learning
#### Clustering Algorithms
* Grouping similar data into separate categories.

#### Non-clustering Algorithms 
* 2 different microphones recording 2 speakers, but intensity of noise varies. 
* Goal: separate the audio.
* Simple implementation (in octave): good for learning ML, prototyping ML
  algorithms before implementing in some other language. 

### Model Representation 
* Housing Prices 
* Input data often called a "training set."
* Hypothesis function h(): maps from x's to y's.
  * Representing h(): h(x) = theta0 + theta1 * x
  * Linear regression with one variable / univariate linear regression 
  * h(x) should be a "good" predictor for the corresponding value of y.

#### Cost Function for Univariate Linear regression Models
* theta0, theta1 = parameters
  * How to find theta0 & theta1 so it's a good predictor of training set?
    * Choose so that h(x) is close to y for our training examples (x,y) 
    * J(theta0, theta1) = Square Error / Cost function = sum((h(x[i]) - y[i])^2)) / 2 \* size(training set) from i = 1 to size(training set)
  * Cost function measures the accuracy of h(x). The larger J(theta0,theta1)
    is, the higher the error of approximation. 
  * Emphasis: find the values for theta0,theta1 to minimze the errors. This
    results in a hypothesis function that best represents the training set.
    * Remember that for each input to J(theta0, theta1), a unique h(x) function
      is generated, but obviously all but one **best** represents the training
set.
  * Mathematically, plotting the points for different values of thetaX result
    in a higher dimsension figure (just theta1 => quadratic (2D))

#### Countour Plots  
* Visualizes 3D figure
* Ultimately, need software to find the optimal parameter values to minimize
  the cost function J().  

#### Univariate Linear Regression - Gradient Descent
* Optimization algorithm which finds local minimum.
* Algorithm which minimizes some arbitrary function (applicable to cost functions). 
  * Repeat until convergence: theta\_j := (theta\_j - (learning\_rate \* (partial\_deriv(theta0, theta1, ...) w.r.t. theta\_j)))
    * Learning rate: constant controlling intensity of descent for each iteration of algorithm
    * Direction of descent is the partial derivative term of the equation.
    * Do this for each parameter "j", **simultaneously**.
      * Computation is done for all parameters, then assignment is done for all parameters.
    * Depending on the starting point in the surface, a different minima could be reached.
* Partial deriv. notation for 2+ parameters, can use derivative notation for a single variable (univariate linear regression models)
* If learning\_rate is too small, gradient descent is slow.
* If learning\_rate is too large, gradient descent can overshoot, causing algorithm to diverge or fail to converge.
* From the gradient descent algorithm, it is clear to see once an optimum is reached, the next iteration won't change the parameter location (hence the convergence property for a gradient descent which doesn't have a "destructive" learning\_rate).
* Furthermore, each g.d. iteration automatically takes smaller steps. The partial deriv. of J() decreases in magnitude, eventually converging to 0.
  * **So, no need to decrease learning rate every iteration.**

#### Gradient Descent Applied to Cost Function:
* Goal: minimizes the squared error through linear regression.
* Cost function for linear regression is a convex function (bowl-shaped surface, so only a global minimum / can only converge to the global minimum) 
* This specific implementation of gradient descent is also called "batch" gradient descent. 
  * **Batch gradient descent):** each step of gradient descent uses *all* training examples (all samples in training set). 
* TTK : Normal equations method: while it doesn't scale as well as gradient descent, this equation calculates the optimal hypothesis function without having to do it iteratively.
