# Neural_Network_From_Scratch

**Neural Network Implementation from Scratch**

**Objective :** Implement a simple feedforward neural network from scratch in Python without using any in-built deep learning libraries. This implementation will focus on basic components like forward pass, backward propagation (backpropagation), and training using gradient descent.

**Problem Definition**

**Dataset :** We'll use the Iris dataset, a famous dataset with 150 samples of iris flowers classified into 3 species.

**Task:** The task is a multi-class classification problem, where we will classify the flowers into one of three species based on the features (sepal length, sepal width, petal length, and petal width).

**Neural Network Architecture :**

**Input Layer :** 4 neurons (since there are 4 features).

**Hidden Layer :** 10 neurons (this is a hyperparameter).

**Output Layer :** 3 neurons (since we have 3 classes).

**Methodology** :

* **Activation Functions :**

 a. **Hidden layer:** ReLU (Rectified Linear Unit)

 b. **Output layer:** Softmax (for multi-class classification).

 c. **Loss Function:** Cross-Entropy Loss (because it's a classification problem).


 * **Optimization:** **Gradient Descent** (we'll use batch gradient descent here).

* **Data Preprocessig:**

 a. We load the Iris dataset using **sklearn.datasets.load_iris**.

 b. Labels are one-hot encoded using **OneHotEncoder** because we have multiple classes.

 c. Features are standardized using **StandardScale**r to improve the training performance.

* **Neural Network Initialization:** We define the number of neurons in each layer and initialize the weights (W1, W2) and biases (b1, b2) with small random values.

* **Activation Functions:**

 a. **ReLU** is used for the hidden layer to introduce non-linearity.

 b. **Softmax** is used in the output layer for multi-class classification.

* **Loss Function:** We use Cross-Entropy Loss which is appropriate for multi-class classification tasks.

* **Gradient Descent (Backpropagation):** We compute the gradients of the loss with respect to the weights and biases and update the parameters using gradient descent.

* **Training:** We train the model for 1000 epochs (you can adjust this) and print the loss every 100 epochs.
 
* **Evaluation:** After training, we evaluate the model on both the training and test sets by computing 
accuracy
