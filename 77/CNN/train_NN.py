import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys
import time
from numpy.lib.stride_tricks import as_strided

    ### "Res" -> {Weight}
    ### "Pool" -> {"Max" or "Avg", Global (bool)}

class CNN_():
    def __init__(self, structure, img_shape):
        
        ### Initialize model weights to 0
        self.structure = structure
        self.img_channels = img_shape[0]
        self.img_shape = img_shape
        self.Architecture, self.Output_Channels = self.construct_layers()

        return

    def construct_layers(self):
        layers = []
        self.input_channels = self.img_channels
        #output_size = self.img_shape
        for idx, layer in enumerate(self.structure):    

            Cin = self.input_channels

            layer_type, params = layer[0], layer[1]

            if layer_type == "Conv":
                layers.append(Conv(params, Cin))
            elif layer_type == "Res":
                continue
            elif layer_type =="Pool":
                layers.append(Pool(params, Cin))

            self.input_channels = params["channels"]

        ### Return the architecture and the shape of the output for softmax
                ## (Do I need to enforce that the last Cout = Y?)
        return layers, self.input_channels
    def first_pass(self, test_img, FC_params):
        # Run a forward pass to get the output shape
        output_shape = self.forward_pass(test_img, first_pass=True)
        # Calculate the total number of features for the FC layer
        num_features = np.prod(output_shape)  # Product of dimensions

        # Initialize FC layer with calculated number of features
        FC_layer = FC(FC_params, num_features)
        self.Architecture.append(FC_layer)

    
    def forward_pass(self, img, first_pass = False):

        output = self.Architecture[0].forward_(img)
        for layer in self.Architecture[1:]:
            if isinstance(layer, FC):
                output = output.reshape(output.shape[0], -1)
                output = layer.forward_(output)
            else:
                output = layer.forward_(output)

        probs = self.softmax_(output)

        if first_pass:
            return output.shape
        
        return probs
            

    def softmax_(self, logits):
        expl = np.exp(logits - np.max(logits, axis = 1, keepdims=True))
        probs = expl / np.sum(expl, axis = 1, keepdims=True)

        return probs
    
    def predict(self, y_pred, y_labels):

        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        loss = -np.sum(y_labels * np.log(y_pred))

        predictions = np.argmax(y_pred, axis = 1)

        return loss, predictions

    def backwards_pass(self, outputs, Y_batch):
        

        ### First, CE Loss
        ### Then, FC
        ### Then, Pooling -> Activation -> Conv xN
        # https://medium.com/@ngocson2vn/a-gentle-explanation-of-backpropagation-in-convolutional-neural-network-cnn-1a70abff508b #
        ### I want to implement batch SGD here.
        for idx, (output, Y) in enumerate(zip(outputs, Y_batch)):


        ### Loss of the fully connected layer 
            print(f"Shape of Y is {Y.shape}")
            label = np.argmax(Y)

            FC_layer = self.Architecture[-1]
                #self.weights.shape
                #(4096, 10)  
            w = FC_layer.weights
                #(10, 4096)
            ### f is the input into the FC layer after max pooling
            ### shape is 64 x 4096, indexed by idx to select the correct row
            f = FC_layer.f[idx]

            dLS = output
            dLS[label] = output[label] - 1
            dLB = np.copy(dLS)
            FC_layer.dLB_ = dLB
            ### 10, 4096 partial derivatives matrix
            dLw = np.zeros((Y.shape[0], f.shape[0]))
            for i in range(dLw.shape[0]):
                dLw[i, :] = output[i] * f[:]
            dLw[i, :] = (output[i] - 1 ) * f[:]
            FC_layer.dLw_ = dLw
            ### For the next step of prop
            dLf = np.zeros(f.shape, dtype=np.float64)
            for j in range(f.shape[0]):
                dLf[j] = np.sum( dLS* w[j, :])
            FC_layer.dLf_ = dLf
            ### DLF gets fed into the previous layer
            ## Placeholder for backprop and data. these can be updated automatically
            ### Everything breaks if my shapes mismatch! let's hope *thaaaat* doesn't happen
            d_F = dLf
            out_ = f

            ### This needs to propagate backwards
            for z, layer_obj in enumerate(reversed(self.Architecture[:-1])):
                if isinstance(layer_obj, Pool):
                    dLP = d_F.reshape(layer_obj.pooled[idx].shape)
                    layer_obj.dLP = dLP
                    d_F = dLP
                    out_ = layer_obj.pooled[idx]
                    I2_ = layer_obj.I2
                elif isinstance(layer_obj, Conv):
                    dLC = np.zeros(layer_obj.activated[idx].shape)
                    # or dLC = np.zeros(layer_obj.layer_output[idx])?
                    for m in range(out_.shape[0]):
                        for i in range(out_.shape[1]):
                            for j in range(out_.shape[2]):

                            
                        #### Need to implement I2 in the max pooling layer

                                um, vm = I2_[idx, m, i, j]
                                dLC[m, um, vm] = d_F[m, i, j]

                    dLS = dLC * layer_obj.dCS
                    dLb = np.zeros(layer_obj.filters)
                    dLk = np.zeros((layer_obj.filters, layer_obj.Cin, layer_obj.kernel_size[0], layer_obj.kernel_size[1]))
                    for m in range(layer_obj.filters):
                        dLb[m] = np.sum(dLS[m])
                        for m in range(layer_obj.Cin): 
                            for p in range(layer_obj.kernel_size[0]):
                                for q in range(layer_obj.kernel_size[1]):
                                    P = 
                                    dLk[m,n,p,q] = np.sum(dLS[m] * )
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
        self.activation = params.get("activation", "relu")

    #["Conv", {"channels":16, "kernel":(3,3), "stride": 1, "pad": 1}],

        ### Initializing the kernel
        self.weights = self.Kaiming_init()
        ### Need to init bias as well probably
        self.biases = np.zeros(self.filters)
    #(Cout, Cin, k_h, k_w)
    def Kaiming_init(self):
        fan_in = self.kernel_size[0]*self.kernel_size[1]*self.Cin
        stdev = np.sqrt(2.0/fan_in)
        #weights_shape = (self.Cout, self.Cin, self.kernel_size[0], self.kernel_size[1])
        weights_shape = (self.Cout, self.Cin * self.kernel_size[0] * self.kernel_size[1])
        weights = np.random.normal(0, stdev, weights_shape)
        return weights
    ### I can reshape back if needed, but it seems like there's no need
    
    def reshaped_kernel(self):
        flat_weights = self.weights.reshape(self.Cout, -1)
        return flat_weights

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

class Residual_FF():
    def __init__(self, params):
        return

class Pool():
    def __init__(self, params, Cin):
        self.channels = params["channels"]
        self.type = params["type"] 
        self.kernel_size = params.get("kernel_size", (2, 2))
        self.global_ = params["global"]
        self.stride = params.get("stride", 2)
        self.Cin = Cin
        self.activation = params["activation"]


    def forward_(self, data):

        if self.global_:
            self.kernel_size = (data.shape[2], data.shape[3])
            self.stride = self.kernel_size

        N, C, H, W = data.shape
        out_h = (H - self.kernel_size[0]) // self.stride + 1
        out_w = (W - self.kernel_size[1]) // self.stride + 1

        #pooled_out = np.zeros((N, C, out_h, out_w))

        strided = as_strided(data, 
                             shape=(N, self.channels, out_h, out_w, self.kernel_size[0], self.kernel_size[1]),
                             strides=(*data.strides[:2], data.strides[2]*self.stride, data.strides[3]*self.stride, *data.strides[2:]),
                             writeable=False)
        
        #I2 = np.zeros((N, C, out_h, out_w, 2), dtype=int)
        
        if self.type == "avg":
            pooled= np.mean(strided, axis=(-2, -1))
        elif self.type == "max":
            max_indices = np.argmax(strided.reshape(N, self.channels, out_h, out_w, -1), axis=-1)
            u, v = np.unravel_index(max_indices, self.kernel_size)
            pooled = np.max(strided, axis=(-2, -1))
            self.I2 = np.stack((u, v), axis=-1)

        #self.I2 = I2
        self.pooled = pooled
        return pooled
    
    def backprop_(self):
        return

class FC():
    def __init__(self, params, Cin):
        self.init_weight = params["init_weight"]
        self.channels = params["channels"]
        self.Cin = Cin
        ### might need to fix the initialization of the FC network.
        ### But, it's right next to the loss function, so maybe not?
        self.weights = np.random.randn(self.Cin, self.channels) * self.init_weight
        self.bias = np.zeros(self.channels)

    def forward_(self, data):
        self.f = data
        self.output = np.dot(data, self.weights) + self.bias
        return self.output



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

    structure = [
        ["Conv", {"activation":"relu", "channels": 8, "kernel":(3,3), "stride": 1, "padding": 1}],
        ["Pool", {"activation":"", "channels": 4, "kernel_size": (2,2), "type": "max", "global":False, "stride":2}],
        ["Conv", {"activation":"relu", "channels":32, "kernel":(3,3), "stride": 1, "padding": 1}],
        ["Pool", {"activation":"", "channels": 4, "kernel_size": (2,2), "type": "max", "global":False, "stride":2}],
        ["Conv", {"activation":"relu", "channels":64, "kernel":(2,2), "stride": 1, "padding": 1}],
        ["Pool", {"activation":"", "channels": 4, "kernel_size": (2,2), "type": "max", "global":False, "stride":2}]
        
        ]
                
    ### Init the model
    ### Features.shape[1] is the number of channels
    img_shape = features.shape[1], features.shape[2], features.shape[3]
    model = CNN_(structure, img_shape)

    test_image = np.zeros((1, img_shape[0], img_shape[1], img_shape[2]))
    ### Initializing the fully connected layer after determining output shape
    model.first_pass(test_image, {"activation":"softmax", "channels": labels.shape[1], "init_weight":0.01})
    
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
            

