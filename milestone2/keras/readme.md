## Data Processing
### 1. Data Processing

* Firstly, we use the 'one-hot' way to encode 'soil_type' data into 0-1 value for them.

* Secondly, for 'wilderness_area' and 'cover_type', we map those string types into int type data. Then we use 'one-hot' encoding to represent this data.

* Reason: We find that the field 'Soid_type', 'wilderness_area' and 'cover_type' is categorical data, and we should not use numerical value to represent character for fire points.

### 2. Model Selection
* We tried to use 'Random Forest' and 'Neural Network'. Finally, we think 'Neural Network' has better performance on this dataset. 
* For neural network, we used 'drop out' on hidden layers, but it is not so good as without using them. And we set 4000 neurons on 4 hidden layers with about 95.3% accuracy on test dataset and 99.2% on training dataset.