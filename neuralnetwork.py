import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("C:\\Users\\jakep\\Desktop\\PythonProjects\\Neural Network From Scratch Python Project\\clustered_dataset.csv")

# plt.scatter(dataset['X'], dataset['Y'], c=dataset['Color'], cmap='coolwarm', edgecolor='k')
# plt.title("Data")
# plt.xlabel("X")
# plt.ylabel("Y")

# plt.show()

x = dataset[["X" , "Y"]].values
y = dataset["Color"].values

# data_array = np.column_stack((X,Y))
# np.random.shuffle(data_array)

# x = data_array[:, :2]
# y = data_array[:, 2]

def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return np.where(z > 0, 1, 0)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis = 1, keepdims = True))
    return exp_z / exp_z.sum(axis = 1, keepdims = True)

def cross_entropy_loss(y_true, y_pred):
    y_true = y_true.astype(int)
    m = y_true.shape[0]
    log_likelihood = -np.log(y_pred[range(m), y_true])
    return np.sum(log_likelihood) / m


def one_hot(y, num_classes):
    y = y.astype(int)
    one_hot_encoded = np.zeros((y.size, num_classes))
    one_hot_encoded[np.arange(y.size), y] = 1
    return one_hot_encoded

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.w1 = np.random.randn(input_size,hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        

    def forward(self, x):
        self.z1 = np.dot(x, self.w1) + self.b1
        self.a1 = relu(self.z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = softmax(self.z2)
        return self.a2

    def backward(self, x, y, learning_rate):
        m = x.shape[0]


        #gets one hot of set y with 3 0's, ex. 001 = 1, 010 = 2, 100 = 3, 
        one_hot_y = one_hot(y,3)
        dz2 = self.a2 - one_hot_y
        dw2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis = 0, keepdims=True) / m

        da1 = np.dot(dz2, self.w2.T)
        dz1 = da1 * relu_derivative(self.z1)
        dw1 = np.dot(x.T, dz1) / m
        db1 = np.sum(dz1, axis = 0, keepdims = True)

        self.w1 -= learning_rate * dw1
        self.b1 -= learning_rate * db1
        self.w2 -= learning_rate * dw2
        self.b2 -= learning_rate * db2

    def loss_function(self, y_true):
        return cross_entropy_loss(y_true, self.a2)


nn = NeuralNetwork(input_size = 2, hidden_size = 10, output_size = 3)


epochs = 1000
learning_rate = 0.01

for epoch in range(epochs):
    output = nn.forward(x)
    loss = nn.loss_function(y)
    nn.backward(x,y,learning_rate)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")




def predict(x):
    output = nn.forward(x)
    return np.argmax(output, axis = 1)
predictions = predict(x)

accuracy = np.mean(predictions == y)
print(f"Accuracy: {accuracy * 100: .2f}%")


def draw_neural_net(ax, left, right, bottom, top, layer_sizes):
    '''
    Parameters
    ----------
    ax : matplotlib.axes.AxesSubplot, the plot to draw the neural network on
    left : float, the left boundary of the plot
    right : float, the right boundary of the plot
    bottom : float, the bottom boundary of the plot
    top : float, the top boundary of the plot
    layer_sizes : list of int, list containing the number of neurons in each layer
    '''
    v_spacing = (top - bottom) / float(max(layer_sizes))
    h_spacing = (right - left) / float(len(layer_sizes) - 1)

    # Nodes
    for i, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing * (layer_size - 1) / 2. + (top + bottom) / 2.
        for j in range(layer_size):
            circle = plt.Circle((left + i * h_spacing, layer_top - j * v_spacing), v_spacing / 4.,
                                color='w', ec='k', zorder=4)
            ax.add_artist(circle)

    # Edges
    for i, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing * (layer_size_a - 1) / 2. + (top + bottom) / 2.
        layer_top_b = v_spacing * (layer_size_b - 1) / 2. + (top + bottom) / 2.
        for j in range(layer_size_a):
            for k in range(layer_size_b):
                line = plt.Line2D([left + i * h_spacing, left + (i + 1) * h_spacing],
                                  [layer_top_a - j * v_spacing, layer_top_b - k * v_spacing], c='k')
                ax.add_artist(line)
import numpy as np
import matplotlib.pyplot as plt

# Assume X and y are your data points and labels
# X is a (n_samples, 2) array of x, y coordinates
# y is a (n_samples,) array of color labels (0, 1, 2)

# Neural Network Predict Function
# This should be your forward propagation function that returns predicted class labels (0, 1, or 2)
def predict(X):
    # Forward pass through the neural network
    output = nn.forward(X)
    return np.argmax(output, axis=1)

# Plotting function to visualize decision boundaries and data points
def plot_decision_boundary(x, y, predict_function):
    # Define the boundaries of the plot
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    
    # Create a mesh grid (for plotting decision boundary)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))
    
    # Flatten the grid, run through the predict function, and reshape the results back to grid shape
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = predict_function(grid)
    Z = Z.reshape(xx.shape)
    
    # Create the plot
    plt.contourf(xx, yy, Z, cmap='coolwarm', alpha=0.6)
    
    # Plot the original data points
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap='coolwarm', edgecolor='k')
    plt.title("Neural Network Decision Boundary with Data Points")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

# Visualize the decision boundary
plot_decision_boundary(x, y, predict)
