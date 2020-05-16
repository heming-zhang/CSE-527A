# Application Project 1

## 1. Data Processing
* Firstly, we use the 'one-hot' way to encode 'soil_type' data into 0-1 value for them.
    * Reason: We find that the field "Soid_type" is categorical data, and we should not use numerical value to represent character for fire points.

## 2. Model Selection
* In the begining, we tried the simple linear model, the linear regression in the 'from sklearn.linear_model', and it did not give us a enough good result to break the baseline1.
* After that, we think that maybe we should implement the ridge regression by 'from sklearn.linear_model import Ridge', but it also not so good if we want to do cross validation to implement the process of model selection.
* Finally, we choose the RidgeCV model from 'from sklearn.linear_model import RidgeCV', which makes us break the baseline1.

## 3. Data Exploration
* Our final weight vector for the linear model for those 16 dimensions are : 
[[ 1.85770724e+00 -3.19455549e+00  8.41894816e+01  5.92221883e-02
  -4.88658909e+00  4.03677847e-01  3.71926348e+01 -9.12273132e+00
   2.04906251e+01 -5.34549810e+02  7.00463277e+02 -3.01974363e+02
   1.36060897e+02  1.34667657e+02  2.84296096e+02 -4.18963754e+02]]

* The fisrt 9 dimensions correspond to those original data frame order, and the last 7 dimensions come from the one-hot encode which are categorical data.

* Through this, we find out that some features have postive impact on the output, like the 'Elevation' and 'Slope', which makes senese.