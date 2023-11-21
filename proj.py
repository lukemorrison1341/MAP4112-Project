import numpy as np
import random
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
def f(x): #Function, y = sin(x)
    return np.sin(x)

class network:
    def __init__(self,n_layers,learning_rate,n_features,n_outputs,density=5,activation=0):
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.density = density
        self.activation = activation #0 = ReLU, 1 = Sigmoid
        layers = []
        for _ in range(self.n_layers):
            if(_ == 0): #Input layer
                layers.append(layer(n_features,isInputLayer=True,density=density))
            elif(_ == (self.n_layers-1)): #Output layer
                layers.append(layer(n_outputs,isOutputLayer=True,density=density))
            else:
                layers.append(layer(density,density=density))
        self.layers = layers
    def fully_connect(self): #Create a dense network, each neuron in a layer connects to every neuron in the next layer. 
        for l in range(self.n_layers): 
            curr_layer = self.layers[l]
            for n in range(curr_layer.n_nodes):
                curr_node = curr_layer.nodes[n]
                if (l == 0): #At the first layer.
                    curr_node.prev_connection = None
                    next_layer = self.layers[l+1]
                    curr_node.connection = next_layer.nodes
                    curr_node.n_connections = len(curr_node.connection)

                elif(l != (self.n_layers - 1)): #If not at the last layer
                        next_layer = self.layers[l+1]
                        curr_node.connection = next_layer.nodes
                        curr_node.n_connections = len(curr_node.connection)
                        prev_layer = self.layers[l-1]
                        curr_node.prev_connection = prev_layer.nodes       
         
                else: #At the last layer
                    curr_node.connection = []
                    prev_layer = self.layers[l-1]
                    curr_node.prev_connection = prev_layer.nodes #Every node in the Output layer's previous connection is all of the second to last layer for a fully connected NN

    def set_weights_bias(self): #Initializes all weights to something random. (Also sets correct shape.)
        for i, l in enumerate(self.layers):
            if(l.isOutputLayer): #Output layer gets identity matrix of shape (n_nodes,n_nodes)
                l.weights_matrix = np.eye(l.n_nodes) #So that input_val @ weights is just input_val, which is just input_val.
                l.bias_vec = np.zeros(l.n_nodes).T
            else: #HE randomization
                next_layer_n_nodes = self.layers[i+1].n_nodes # Number of nodes in the next layer.
                curr_nodes = l.n_nodes
                stddev = np.sqrt(2 / curr_nodes)

                l.weights_matrix = np.random.normal(0,stddev,(next_layer_n_nodes,curr_nodes))
                l.bias_vec = np.random.rand(next_layer_n_nodes).T

    def forward_pass(self,input):
         # Assert that the input data shape matches n_features
        if isinstance(input,list):
            input = np.array(input)
            assert len(input) == self.n_features, "Input Data Shape does not match the n_features."
        else: #Is already numpy array 
            assert input.size == self.n_features, "Input Data Shape does not match the n_features."
        #Perform forward pass.
        input = input.T
        for i, l in enumerate(self.layers):
            l.input_vec = input
            if(l.isOutputLayer):
                return val
            elif(l.isInputLayer):
                #print("Applying no activation ...")
                val = l.forward(activation=-1)  
            else:
                val = l.forward(self.activation)
             
            input = val


    
    def ReLU(vec):
        new_vec = []
        for x in vec:
            if(x < 0):
                new_vec.append(0.01*x)
            else:
                new_vec.append(x)
        new_vec = np.array(new_vec)
        return new_vec.T
        
    def Sigmoid(vec):
        #print("USING SIGMOID")
        vec = np.clip(vec, -50, 50) #Prevent NaN values from np.exp
        new_vec = []
        for x in vec:
            new_vec.append( 1/ (1+np.exp(-x)))
        new_vec = np.array(new_vec)
        return new_vec.T
    def activation_function(vec,activation=0): #Using ReLU activation function or sigmoid
        if(activation == 0):
            return network.ReLU(vec)
        else:
            return network.Sigmoid(vec)
    
    def backpropogate(self,output,target): #dC/dW = (dZ/dW) * (dA/dZ) * (dC/dA), Z = unactivated output, A = activated output, C = cost function, W = weights matrix
        dC_dA = self.derivative_cost_respect_to_activation(output,target) #There was a loss associated with output and target. How much does the cost change with respect to the activated value ? 
        #print("Output:",output)
        for i,l in enumerate(reversed(self.layers)): #Start at the Last layer, go up until the very firsy layer.
            if(l.isOutputLayer): continue

            if(l.input_vec.shape == ()):
                z = (l.weights_matrix * l.input_vec).flatten() + l.bias_vec #Flatten to treat it as row-vector. 
            else:
                z = l.weights_matrix @ l.input_vec + l.bias_vec
            
            #print("Value of dC_dA:",dC_dA)
            activation_prev = l.input_vec

            #Compute the partial derivatives and then compute dC/dW in one line.

            dC_dZ = dC_dA * self.derivative_activation_respect_to_z(z) #Element-wise multiplication. Is dC/dZ. 
            #print("Value of dC_dZ:",dC_dZ)
            #print("Value of dA/dZ",self.derivative_activation_respect_to_z(z))
            #print("Value of dZ_dW",self.derivative_z_respect_to_weights(activation_prev,z))

            dC_dW = np.outer(dC_dZ,self.derivative_z_respect_to_weights(activation_prev,z)) #
            dB = dC_dZ

            l.weights_gradient = dC_dW
            l.bias_gradient = dB

            #Update dC_dA for next layer
            dC_dA = l.weights_matrix.T @ dC_dZ
            #print("Shape of new gradient:",gradient.shape)
            #print("SHAPE OF WEIGHTS gradient:",l.weights_gradient.shape)
            #print("\n")
    def derivative_activation_respect_to_z(self,vec):
        
        if(self.activation == 0):
            new_vec = []
            for x in vec:
                if(x < 0):
                    new_vec.append(0.01)
                else:
                    new_vec.append(1)
            new_vec = np.array(new_vec)
            return new_vec.T
        else:
            new_vec = []
            for x in vec:
                new_vec.append(x * (1-x))
            new_vec = np.array(new_vec)
            return new_vec
    
    def derivative_cost_respect_to_activation(self,output,target):
        return ((2 / len(output)) * (output - target))

    def derivative_z_respect_to_weights(self,activation_prev,z): #Is just activation_prev in this case.
        return activation_prev

    def loss(self,output,target): #MSE 
        return np.mean(output - target) ** 2
    def update_weights(self): #Gradient Descent.
        for i, layer in enumerate(reversed(self.layers)):
            if(layer.isOutputLayer): continue
            layer.weights_matrix = layer.weights_matrix - (self.learning_rate * layer.weights_gradient)
            layer.bias_vec = layer.bias_vec - (self.learning_rate * layer.bias_gradient)




    def print_network(self):
        x = 1
        for l in self.layers:
            print("Layer :",x)
            l.print() 
            x = x + 1
                    
class layer(network):
    def __init__(self,n_nodes,isInputLayer=False,isOutputLayer=False,density=5):
        self.n_nodes = n_nodes
        self.isInputLayer = isInputLayer
        self.isOutputLayer = isOutputLayer
        self.input_vec = None 
        self.bias_vec = None
        self.bias_gradient = None
        nodes = []
        if(isOutputLayer): #n_weights of each node should be equal to network density
            for _ in range(self.n_nodes):
                nodes.append(node()) #Output layer doesn't get any weights.
        else: #All other layers
            for _ in range(self.n_nodes):
                nodes.append(node()) 
        self.nodes = nodes
        self.weights_matrix = None
        self.weights_gradient = None
                
    def print(self):
        for n in self.nodes:
            n.print()
    def forward(self,activation=0):#Move weights @ input_vec to next layers input_vec, perform network's activation function.
        #print("Shape of weights:",self.weights_matrix.shape, "Shape of input_vec",self.input_vec.shape, "Shape of bias",self.bias_vec.shape)
        if(activation != -1):
            if(self.input_vec.shape == ()):
                vector = (self.weights_matrix * self.input_vec).flatten() + self.bias_vec #Flatten to treat it as row-vector. 
                activation = network.activation_function(vector,activation=activation)
            else:
                vector = (self.weights_matrix @ self.input_vec) + self.bias_vec
                activation = network.activation_function(vector,activation=activation)
        else: # Don't apply an activation function (first layer condition)
            if(self.input_vec.shape == ()):
                activation = (self.weights_matrix * self.input_vec).flatten() + self.bias_vec #Flatten to treat it as row-vector. 
            else:
                activation = (self.weights_matrix @ self.input_vec) + self.bias_vec

        return activation




class node(layer):
    def __init__(self):
        self.value = 0 #Value of the nodes individual contribution, that will be sent to all nodes it is connected to as input_val. This is BEFORE activation. 
        self.derivative = 0
        self.input_val = [] #Was value of all connections summed up before this. Is a list, list of all values from connections. Needs to be summed and then activated to form a value for the connecting nodes
        self.connection = []
        self.prev_connection = []
        self.n_connections = 0
        self.weights = []
    def print(self):
        print("Node!",self.n_connections," connections")

iris = load_iris()
X, y = iris.data, iris.target

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.5, random_state=42)



epochs = 12
n = network(n_features=4,n_outputs=1,density=6,learning_rate=0.01,n_layers=7)
n.fully_connect()
n.set_weights_bias()
loss_per_val = []
loss_per_epoch = []
zeroing = False
for epoch in range(epochs):
    for i,x in enumerate(X_train):
        outp = n.forward_pass(x)
        loss = n.loss(outp,y_train[i])
        loss_per_val.append(loss)
        n.backpropogate(outp,y_train[i])
        n.update_weights()
    loss_per_epoch.append(sum(loss_per_val) / len(loss_per_val))
    if(epoch != 0):
        prev_epoch_loss = loss_per_epoch[epoch-1]
    if(epoch == 1):
        if(prev_epoch_loss - loss_per_epoch[epoch] < 0.15): # It's zeroing out.
            print("Zeroing")
            zeroing = True
            break
    loss_per_val = []
print(loss_per_epoch)

if(zeroing):
    for l in n.layers:
        print("Weights :",l.weights_matrix,"\nWeights gradient:",l.weights_gradient, "Bias:",l.bias_vec,"Bias gradient:",l.bias_gradient)

test_answers = []
for i, x in enumerate(X_test):
    outp = n.forward_pass(x)
    test_answers.append(outp)
plt.xlabel("Samples")
plt.ylabel("Value")


plt.plot(test_answers,label="Model prediction",linestyle=':')
plt.plot(y_test,label="True value")
plt.legend()
plt.show()