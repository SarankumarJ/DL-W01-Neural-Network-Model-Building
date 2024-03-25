# DL-W01-Neural-Network-Model-Building
### Name : Sarankumar J
### Reg No : 212221230087
## Program
```py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the dataset
data = pd.read_csv("mushrooms.csv")

# Separate features (X) and target variable (y)
X = data.drop('class', axis=1)
y = data['class']

# Convert categorical input to numeric values using LabelEncoder
label_encoders = {}
for column in X.columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le

# Convert the categorical output to numeric values
y = LabelEncoder().fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a TensorFlow model with appropriate activation functions and number of neurons in the output layer
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=15, batch_size=64, validation_data=(X_test, y_test))

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)
```

## Output

### Draw the neural network architecture for your model using the following website

![image](https://github.com/SarankumarJ/DL-W01-Neural-Network-Model-Building/assets/94778101/f37e6aa7-d259-43cf-95d5-ca244cdbf541)

![image](https://github.com/SarankumarJ/DL-W01-Neural-Network-Model-Building/assets/94778101/138956c6-73d1-4108-a0c6-ae3b1fa00e10)

### Github URL
https://github.com/SarankumarJ/DL-W01-Neural-Network-Model-Building/
