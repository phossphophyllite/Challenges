import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys



    ### "Res" -> {Weight}
    ### "Pool" -> {"Max" or "Avg", Global (bool)}

class CNN_():
    def __init__(self, structure, img_channels):
        
        ### Initialize model weights to 0
        self.structure = structure
        self.img_channels = img_channels

        self.Architecture, self.Output_Channels = self.construct_layers()

        return
    def construct_layers(self):
        layers = []
        self.channels_out = 1
        for idx, layer in enumerate(self.structure):                
            if idx == 0:
                Cin = self.img_channels
            else:
                Cin = self.channels_out

            layer_type, params = layer[0], layer[1]
            ### Update channels out for the next layer.
            ### This is going to be a bit more difficult to handle
            self.channels_out = params["channels"]
            
            if layer_type == "Conv":
                layers.append(self.Conv(params, Cin))
            if layer_type == "Res":
                layers.append(self.Residual_FF(params, Cin))
            if layer_type =="Pool":
                layers.append(self.Pool(params, Cin))
        ### Return the architecture and the shape of the output for softmax
                ## (Do I need to enforce that the last Cout = Y?)
        return layers, self.channels_out

    def softmax_(self):
        return
    def ReLU(self):
        return


### "Conv" -> { Filters (Cout), (Kernel Size), stride, padding, img channels (Cin)}
class Conv():
    def __init__(self, params, Cin):
        self.filters = params["channels"]
        self.Cout = self.filters
        self.kernel_size = params["kernel"]
        self.stride = params["stride"]
        self.Cin = Cin
        ### I think I need to determine the padding programmatically
        ### to make sure H + p / s, W + p / s = integer
        self.padding = params["padding"]

    #["Conv", {"channels":16, "kernel":(3,3), "stride": 1, "pad": 1}],

        ### Initializing the kernel
        self.weights = self.Kaiming_init()

    def Kaiming_init(self):
        fan_in = self.kernel_size[0]*self.kernel_size[1]*self.Cin
        stdev = np.sqrt(2.0/fan_in)
        weights_shape = (self.Cout, self.Cin, self.kernel_size[0], self.kernel_size[1])
        weights = np.random.normal(0, stdev, weights_shape)
        return weights
    
    def im2col(self, data):

        ### dimensions are N x C x h+p x w+p
        N, C, H, W = data.shape
        k_h, k_w = self.kernel_size
        stride = self.stride

        ### Padding has already been applied to the data.
        out_h = (H - k_h) // stride + 1
        out_w = (W - k_w) // stride + 1

        col = np.zeros((N, C, k_h, k_w, out_h, out_w))

        for y in range(k_h):
            y_max = y + stride * out_h
            for x in range(k_w):
                x_max = x + stride*out_w
                col[:, :, y, x, :, :] = data[:, :, y:y_max:stride, x:x_max:stride]
        ### (N, out_h, out_w, C, k_h, k_w)
        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, C*k_h*k_w)
        print(f"After im2col, data shape is {col.shape} = N*out_h*out_w, C*k_h*k_w")
        return col
    
    def forward_(self, data, img_shape, output_size):
        ### Once the full layer list has been initialized, we can
        ### assign the output size of the following layers
        self.output_size = output_size
        shape = self.kernel_size
        stride = self.stride

        y_padding = ((output_size[0] - 1) * stride - img_shape[0] + self.kernel_size[0])
        y_padding = max(0, y_padding) 
        x_padding = ((output_size[1] - 1) * stride - img_shape[1] + self.kernel_size[1])
        x_padding = max(0, x_padding)

        top_padding = y_padding // 2
        bottom_padding = y_padding // 2 + y_padding % 2
        left_padding = x_padding // 2
        right_padding = x_padding // 2 + x_padding % 2
        ypad = (top_padding, bottom_padding)
        xpad = (left_padding, right_padding)

        batch_size = data.shape[0]    
        ### Shape for a batch should be batch_size, image_channels, flattened_length
        ### For each layer I'll need to also output the img_shape tuple to unflatten, or I could just reshape from the start

        ### Data is currently flattened in dimension 3, Reshaping for padding


        data_padded = np.pad(data, ((0, 0), (0, 0), (ypad[0], ypad[1]), (xpad[0], xpad[1])), mode='constant', constant_values=0)


        ### Moving to im2col
        
        reshaped_im = self.im2col(data_padded)
                
        ### Feedforward time!!!
        

        return 
    

    def backprop_(self):
        return
    
    ### Full pre-activation:
    # In -> Batch norm -> ReLU -> Weight -> BN -> ReLU -> Weight -> addition

class Residual_FF():
    def __init__(self, params):
        return
    def BN(self):
        return
    def weight(self):
        return
    def ReLU(self):
        return



class Pool():
    def __init__(self, params):
        return
    def forward_(self):
        return
    def backprop_(self):
        return

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

    structure = [

        ["Conv", {"channels":16, "kernel":(3,3), "stride": 1, "pad": 1}],
        ["Res", {"channels": 16, "block_layers": 2, "stride": 1, "expansion": 4}],
        ["Pool", {"channels": 4, "type": "max", "global":False}]
        # ,
        # ["Block", {"Structure":
        # }]
        ]
                
    ### Init the model
    ### Features.shape[1] is the number of channels
    model = CNN_(structure, features.shape[1])

    ### For each step, I split into train/validation sets.
    ### Pass validation sets into the class, train, 
    ### then test against validation, for each epoch


    for e in range(epochs):
    ### Split the validation set off here

        for X_batch, Y_batch in model.gen_batches(features, labels, batch_size):
            print()

    return model
            

