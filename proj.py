import numpy as np
import random
from matplotlib import pyplot as plt
class network:
    def __init__(self,n_layers,learning_rate,n_features,n_outputs,density=5):
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.density = density
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

    def set_weights(self): #Initializes all weights to something random. (Also sets correct shape.)
        for l in self.layers:
            if(isInputLayer):
                next_layer_n_nodes = self.layers[]
            elif(isOutputLayer):

            else:


    def forward_pass(self,input):
         # Assert that the input data shape matches n_features
        if isinstance(input,list):
            input = np.array(input)
        assert len(input) == self.n_features, "Input Data Shape does not match the n_features."      
        #Perform forward pass.
        for l in self.layers:
            print(l.weights_matrix)


    def activation_function(x): #Using ReLU activation function.
        if(x < 0):
            return 0
        else:
            return x


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
        nodes = []
        if(isOutputLayer): #n_weights of each node should be equal to network density
            for _ in range(self.n_nodes):
                nodes.append(node()) #Output layer doesn't get any weights.
        else: #All other layers
            for _ in range(self.n_nodes):
                nodes.append(node()) 
        self.nodes = nodes
        self.weights_matrix = None
                
    def print(self):
        for n in self.nodes:
            n.print()
    def forward(self):#All value attributes of this layer, is input Value to next layer's connections. Perform activation function. 
        current_val = 0
        for n in self.nodes:
            current_val = n.value
            for c in n.connection:
                c.input_val.append(network.activation_function(current_val))
    def sum_inputs(self):   #Sum up all of the input_val of a layer, places it back into input_val temporarily 
        for n in self.nodes:
            n.input_val = sum(n.input_val)
    def get_value_vector(self): #Put each node's value from a layer in a vector. Used for calculating wX
        value_vec = []
        for n in self.nodes:
            value_vec.append(n.value)
        value_vec = np.array(value_vec)
        return value_vec.T




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


n = network(n_layers=4,learning_rate=0.1,n_features=2,n_outputs=2,density=3)
n.fully_connect()
n.print_network()
test_input = [1.3,2.8]
output = n.forward_pass(test_input)