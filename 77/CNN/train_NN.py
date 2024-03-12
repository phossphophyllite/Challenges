from activations import activation_functions_, activation_derivatives_
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys
import time
from numpy.lib.stride_tricks import as_strided
from collections import OrderedDict

    ### "Res" -> {Weight}
    ### "Pool" -> {"Max" or "Avg", Global (bool)}

class Model():
    def __init__(self, Architecture):
        
        ### Initialize model weights to 0
        self.Architecture = Architecture

        return
    
    def forward_pass(self, img, first_pass = False):


        ### Use adaptive pooling between end-1 layer and end layer

        return probs
            

    def softmax_(self, logits):

        return probs
    
    def predict(self, y_pred, y_labels):

        return loss, predictions

    def backwards_pass(self, outputs, Y_batch):
        
        return

class CNN_Architecture():
    def __init__(self, layers, img_channels):

        ### I use this during forward, backward pass
        ### Or do I? hehehehe
        self.reg = {"start": None, "end": None}
        self.Structure = self.Create_Structure(layers, img_channels)


    def Create_Structure(self, layers, img_channels):  
        Structure = {}
        input_shape = img_channels

        ### Iterating through the layers in the ordered Dict.
        for name, layer in layers.items():
            ### For reference by Model().        
            flag = None
            if layer.get("start") is not None:
                self.reg["start"] = name
            if layer.get("end") is not None:
                self.reg["end"] = name

            if layer.get("input") is None:
                print(f"No input layer assigned for layer {name}")
                raise Exception("No input layer assigned.")
            
            ### The input layer corresponding to the key stored at the current layer's "input" item
            input_layer = Structure.get(layer["input"])

            ### I should not have outputs from any layer that doesn't have kernels (pooling and Conv). 
            ### If the input layer is FC but not the end, then the kernel can be set to 1, though this shouldn't realistically happen.
            if input_layer is None:
                print(f"Layer {name} is trying to access layer {layer["input"]}, which does not have an object assigned yet. Check order in the ordered Structure dict.")
                raise Exception("Incorrect input layer reference.")
            if not hasattr(input_layer, 'k'):
                print(f"Layer {name} is trying to access the kernel shape from layer {layer["input"]}, which does not have a kernel assigned yet. Check type assignment in the ordered Structure dict.")
                raise Exception("Incorrect input size assignment.")
            
            ### Generating layers
            layer_obj = self.create_layer(layer, input_shape)
            input_shape = layer_obj.k
            Structure[name] = layer_obj

        return Structure

    def create_layer(self, layer, input_shape):

        layer_input = layer["input"]
        layer_type = layer["type"]
        params = layer["params"]

        ### In general, 'input shape' is just the number of kernels in the preceeding pool/conv layer.
        ### If the layer has a residual connection, this is handled during the residual forward pass, i.e. not by that layer itself.
        if layer_type == 'conv':
            layer_obj = Conv(params, layer_input, input_shape)

        if layer_type == 'pool':
            layer_obj = Pool(params, layer_input, input_shape)

        if layer_type == 'res':
            layer_obj = Res(params, layer_input, input_shape)
    
        if layer_type == 'fc':
            layer_obj = FC(params, layer_input, input_shape)
        
        return layer_obj


class Conv():
    def __init__(self, params, layer_input, input_shape):
        self.layer_input = layer_input ### Previous layer that the input is derived from during forward pass
        self.input_shape = input_shape ### Number of kernels output from the previous layer
        self.k = params["k"] ### Number of kernels
        self.kernel_shape = params["shape"] ### Square dimension kernels assumed throughout, because why wouldn't I?
        self.stride = params["stride"] ### Stride
        self.activation = params["activation"]

        self.init_layer()

    def init_layer(self):
        self.init_weights()

    ### Using Kaiming initialization
    def init_weights(self):
        fan_in = self.kernel_shape * self.kernel_shape * self.input_shape
        stdev = np.sqrt(2.0/fan_in)
        self.shape = (self.k, self.input_shape, self.kernel_shape, self.kernel_shape)
        self.weights = np.random.normal(0, stdev, self.shape)
        self.biases = np.zeros(self.k)

    def flatten(self):
        return self.weights.reshape(self.k, -1)

    def forward_():
        return

    def backward_(self, dz):
        return()
    
class Pool():
    def __init__(self, params, layer_input, input_shape):
        self.layer_input = layer_input ### Previous layer that the input is derived from during forward pass
        self.input_shape = input_shape ### Number of kernels output from the previous layer
        self.k = params["k"] ### Number of kernels
        self.kernel_shape = params["shape"] ### Kernel shape, assumed square
        self.stride = params["stride"] ### Kernel stride
        self.type_ = params["type"] ### MAX or AVG
        self.global_ = params["global"] ### whether global

        self.init_layer()

    ### Doesn't seem like there's anything needed here right now. 
    def init_layer(self):
        return
    
### This is applied to each kernel. Number of kernels doesn't change. This is fine
class Adaptive_Pool():
    def __init__(self, input_shape, output_shape, params):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.type_ = params["type"]
        self.global_ = params["global"]

    ### Run this during execution.
    ### Might be slow, but right now it's only partly vectorized
    def forward_(self, input_):
        
        ### Input shape is of the form (N x Nk x m x n)
        ### Output shape is of the form (N x self.output_shape)
        ### If global, it's easy - we just flatten and max/avg each row
        ### If local, it's not so easy - we run adaptive pooling, and only then flatten

        ### Only use this if the number of kernels in the final layer matches the 
        ### shape of the fully connected layer, defined in Structure["APn"]
        ### i.e. your last conv has 2048 kernels, and the FC layer has 2048 neurons
        if self.global_:

            ### Flatten into (N x Nk x m * n) by condensing the last 2 axes
            flattened = input_.reshape(input_.shape[0], input_.shape[1], -1) 
            self.input_ = input_
            output = np.zeros((input_.shape[0], input_.shape[1]))


            if self.type_ == "AVG":
                ### The gradient is distributed as a fraction, i.e. propagated as
                ### (1 / ( kernel shape ^2 )) * dL/dz 
                ### for each node. This might be easier, since I can just store it at the time.
                output = np.mean(flattened, axis=2)
                return output
            elif self.type_ == "MAX":
                ### Store indices of the 'max' values used as a boolean mask.
                ### Propagate backwards mask * dL/dz
                output = np.max(flattened, axis=2)
                return output

        ### This is performed for each kernel. 
        N, Nk, H, W = input_.shape
        H_prime, W_prime = self.output_shape
        stride_H = int(H / H_prime)
        stride_W = int(W / W_prime)
        self.stride_H = stride_H
        self.stride_W = stride_W
        self.H_prime = H_prime
        self.W_prime = W_prime
        output = np.zeros((N, Nk, H_prime, W_prime))

        for i in range(H_prime):
            for j in range(W_prime):
                start_H = i * stride_H
                start_W = j * stride_W 
                end_H = start_H + stride_H
                end_W = start_W + stride_W
                if self.type == "AVG":
                    output[:, :, i, j] = np.mean(input_[:, :, start_H:end_H, start_W:end_W], axis = (2,3))
                elif self.type == "MAX":
                    output[:, :, i, j] = np.max(input_[:, :, start_H:end_H, start_W:end_W], axis = (2,3))
    def backward_(self, dz):
        return()

class Res():
    def __init__(self, params, layer_input, input_shape):
        self.layer_input = layer_input ### Previous layer that the input is derived from during forward pass
        self.input_shape = input_shape ### Number of kernels output from the previous layer
        self.output_layer = params["output"] ### Target output layer of the residual. During forward pass, output is added to this class.
        self.resize = params["resize"] ### If the shape should be resized. 

    def init_layer(self):
        return

class FC():
    def __init__(self, params, layer_input, input_shape):
        self.layer_input = layer_input ### Input layer (usually a pooling layer)
        self.input_shape = input_shape ### Input shape (kernels)
        ### I'm using adaptive pooling. 
        ### If adaptive pooling isn't being used, you'll need to resize the input somehow else.
        self.labels = params["labels"] ### Number of classes
        self.shape = params["shape"] ### Ideal shape of the matrix
                                     ### In other words, for an input of shape [k * m * n],
                                     ### the matrix needs to be [k * m * n, Y] for Y labels
        self.init_weight = params["init_weight"]
        self.init_layer()

    def init_layer(self):
        self.weights = np.random((self.shape), self.labels) * self.init_weight
        self.bias = np.zeros(self.labels) ### One bias per output channel
        return
    
    def forward_(self, input_):
        self.input_ = input_
        self.output = np.dot(input_, self.weights) + self.bias
        return self.output
    
    def backward_(self, dz):
        return()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    ### Structure: each key is a specific layer, which its own standardized params.
    ### Important to note ensure that the input connections are correct.
    ### Each key stores its parameters

    ### Layer parameters:
                # "type":       layer type, 'conv', 'pool', 'adpool' 'res', 'fc'; more to come :D
                # "input":      which layer this layer grabs its input from. dynamically referenced, 
                #               so it's important to make sure it's correct.
                #               I'm not explicitly handling 'residual blocks', but manually defining connections.
                # "obj":        the class object will be stored here after init. not necessary but 
                #               helps with debugging - can be cut out later.

    ### Optional flags:
                # "start":      whether this is the initial layer (boolean)
                # "end":        whether this is the output layer, usually FC (boolean)

    ### Conv layer params: 
                # "k":          number of convolutional kernels (int)
                # "shape":      kernel w x w, assume square kernels (int)
                # "stride":     stride (int)
                # "activation": activation function, 'relu' or 'tanh' (string)

    ### Pooling layer params:
                # "k":          number of pooling kernels (int)
                # "stride":     stride (int)
                # "type":       whether 'AVG' or 'MAX' (string)
                # "global":     if global pooling (boolean)
    
    ### Adaptive Pooling layer params:
                # "type":       whether 'AVG' or 'MAX' (string)
                # "global":     if global pooling (boolean)
                #               There is no kernel specification, because Nkin = Nkout.

    ### Residual layer params:
                # "output":     name of the layer it's outputting too (string)
                # "resize":     whether the layer should be resizing to match the output layer (boolean)
                #               this could be dynamically determined as a function (if shape is wrong, 
                #               when called, reshape input to output)

    ### Fully connected layer params:
                # "shape":      should be input as a variable determining the shape 
                #               of the OH classification label. i.e. for MNISt, 10. (int)

structure = OrderedDict([
    ("C1", {"type": 'conv', "input": None,
            "params": {"k": 64, "shape": 7, "stride": 2, "activation": 'relu'},
            "obj": None, "start": 1}),

    ("P1", {"type": 'pool', "input": "C1",
            "params": {"k": 12, "stride": 2, "type": 'MAX', "global": 0},
            "obj": None}),

    ("R1", {"type": 'res', "input": "P1",
            "params": {"output": 'C4', "resize": 1},
            "obj": None}),

    ("C2", {"type": 'conv', "input": "P1",
            "params": {"k": 128, "shape": 3, "stride": 1, "activation": 'relu'},
            "obj": None}),

    ("C3", {"type": 'conv', "input": "C2",
            "params": {"k": 128, "shape": 3, "stride": 1, "activation": 'relu'},
            "obj": None}),

    ("C4", {"type": 'conv', "input": "C3",
            "params": {"k": 128, "shape": 3, "stride": 1, "activation": 'relu'},
            "obj": None}),

    ("C5", {"type": 'conv', "input": "C4",
            "params": {"k": 128, "shape": 3, "stride": 1, "activation": 'relu'},
            "obj": None}),

    ("C6", {"type": 'conv', "input": "C5",
            "params": {"k": 128, "shape": 3, "stride": 2, "activation": 'relu'},
            "obj": None}),

    ("C7", {"type": 'conv', "input": "C6",
            "params": {"k": 128, "shape": 3, "stride": 1, "activation": 'relu'},
            "obj": None}),

    ("AP1", {"type": 'adpool', "input": "C7",
            "params": {"type": 'MAX', "global": 0},
            "obj": None}),

    ("FC", {"type": 'fc', "input": "AP1",
            "params": {"labels": Y.shape[1], "shape": 4096, "init_weight": 0.01}, #### Decide Y.shape elsewhere, placeholder
            "obj": None, "end": 1}),
])
# Ensure all names are unique!
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

####
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ OLD ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

### holding on to this for reference
class Conv_():
    def __init__(self, params, Cin):
        return
    ### I can reshape back if needed, but it seems like there's no need

    def im2col(self, data):

        ### dimensions are N x C x h+p x w+p
        N, Cin, H, W = data.shape
        k_h, k_w = self.kernel_size
        stride = self.stride

        ### Padding has already been applied to the data.
        out_h = (H - k_h) // stride + 1
        out_w = (W - k_w) // stride + 1

        col = np.zeros((N, Cin, k_h, k_w, out_h, out_w))

        for y in range(k_h):
            y_max = y + stride * out_h
            for x in range(k_w):
                x_max = x + stride*out_w
                col[:, :, y, x, :, :] = data[:, :, y:y_max:stride, x:x_max:stride]
        ### (N, out_h, out_w, C, k_h, k_w)
        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, Cin*k_h*k_w)
        #print(f"After im2col, data shape is {col.shape} = N*out_h*out_w, C*k_h*k_w")
        return col
    

    def pad_img(self, data):
        output_size = self.output_size
        img_shape = self.img_shape
        stride = self.stride
        y_padding = ((output_size - 1) * stride - img_shape[0] + self.kernel_size[0])
        y_padding = max(0, y_padding) 
        x_padding = ((output_size - 1) * stride - img_shape[1] + self.kernel_size[1])
        x_padding = max(0, x_padding)

        top_padding = y_padding // 2
        bottom_padding = y_padding // 2 + y_padding % 2
        left_padding = x_padding // 2
        right_padding = x_padding // 2 + x_padding % 2
        ypad = (top_padding, bottom_padding)
        xpad = (left_padding, right_padding)
        self.ypad = ypad
        self.xpad = xpad
        data_padded = np.pad(data, ((0, 0), (0, 0), (ypad[0], ypad[1]), (xpad[0], xpad[1])), mode='constant', constant_values=0)
        return data_padded
    
    def out_to_4D(self, out):
        out_h = (self.img_shape[0] + sum(self.ypad) - self.kernel_size[0]) // self.stride + 1
        out_w = (self.img_shape[1] + sum(self.xpad) - self.kernel_size[1]) // self.stride + 1
        ###(2D (N * out_h * out_w, C * k_h * k_w)) -> (N, C, out_h, out_w)
        out_4D = out.reshape(self.batch_size, out_h, out_w, self.Cout).transpose(0,3,1,2)
        return out_4D

    def forward_(self, data):
        ### Once the full layer list has been initialized, we can
        ### assign the output size of the following layers
        self.batch_size = data.shape[0]
        self.output_size = self.filters
        self.img_shape = (data.shape[2], data.shape[3])

        ### Data is currently flattened in dimension 3, Reshaping for padding
        data_padded = self.pad_img(data)
        ### Moving to im2col
        reshaped_im = self.im2col(data_padded)
        ### Reshaping the weights to 2D
        self.weights = self.reshaped_kernel()
        ### Feedforward time
        Out = np.dot(reshaped_im, self.weights.T)

        Out +=self.biases[None, :]

        ### Reshaping output back to 4D
        reshaped_output = self.out_to_4D(Out)
        ### Storing outputs for backpropagation

        self.layer_output = reshaped_output
        #print(f"I made it through forward pass! ")
        #(N, C, out_h, out_w)

        ### Applying activation
        activated_data = self.activation_(reshaped_output)
        ### Derivative of RELU. I'll need to replace this with another lambda function once I implement tanh, lrelu, etc
        self.dCS = np.where(reshaped_output > 0, 1, 0)
        self.reshaped_output = reshaped_output
        self.activated = activated_data
        return reshaped_output

    def activation_(self, data):
        activations = {
            "relu": lambda x: np.maximum(0, x),
            "lrelu": lambda x: np.where(x > 0, x, 0.01 * x),
            "tanh": lambda x: np.tanh(x),
            "logistic": lambda x: 1 / (1 + np.exp(-x))
        }
        act_func = activations.get(self.activation, activations["relu"])
        return act_func(data)
    
    def backprop_(self):
        return
    
    ### Full pre-activation:
    # In -> Batch norm -> ReLU -> Weight -> BN -> ReLU -> Weight -> addition


def gen_batches(features, labels, batch_size):
    N = features.shape[0]
    ### Shuffling for each epoch
    indices = np.random.permutation(N)

    for start_i in range(0, N, batch_size):
        end_i = min(start_i + batch_size, N)
        batch_i = indices[start_i:end_i]
        yield features[batch_i], labels[batch_i]

        
def train_CNN(features, labels, hyperparams):
   
    #(batch_size, learning_rate, validation_split, epochs)
    batch_size = hyperparams[0]
    LR = hyperparams[1]
    split = hyperparams[2]
    epochs = hyperparams[3]



    ### ~~~ This should be moved out to Main.py so I can access it easily.

    ### Generating the layers. "Conv" for a convolution layer (wow!), 
    ### Res for a residual feedforward connect, (using full pre-activation)
    ### Pool for a pooling layer
    ### Block for a resnet-style block. Not sure how I'll
    ### handle this, maybe generating a separate class and just referencing
    ### "Block" within the structure list?
    ### This is in the form of a Nx2 array. First value identifies the layer 
    ### type. 
    ### First value of the 'params' list needs to be the channels, i.e.
    ### for a Conv the number of filters
    ### Last layer needs to be a Fully connected layer that can be mapped to softmax output.
    ### Average and global pooling not enabled.


    
    ### For each step, I split into train/validation sets.
    ### Pass validation sets into the class, train, 
    ### then test against validation, for each epoch

    for e in range(epochs):
    ### Split the validation set off here
        
        batch_count = 1
        for X_batch, Y_batch in gen_batches(features, labels, batch_size):
            batch_count+=1
            probs = model.forward_pass(X_batch)
            CE_loss, predictions = model.predict(probs, Y_batch)
            print(f"Training loss is {CE_loss} for batch {batch_count} out of {int(features.shape[0]/batch_size)}")

            model.backwards_pass(probs, Y_batch)

    return model
            

