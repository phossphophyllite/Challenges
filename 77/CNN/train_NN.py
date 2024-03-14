from activations import activation_functions_, activation_derivatives_
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys
import time
from numpy.lib.stride_tricks import as_strided
from collections import OrderedDict

class Model():
    def __init__(self, Architecture):
        
        self.Architecture = Architecture

        return
    
    def forward_pass(self, X_batch, Y_batch):

        for layer_, layer_obj in self.Architecture.Structure.items():
            
            ### First layer
            if layer_ == self.Architecture.reg["start"]:
                layer_obj.forward_(X_batch)
                continue

            ### Key corresponding to the input layer
            layer_input = layer_obj.layer_input
            ### Output variable (stored as output_) of the input_layer
            if not hasattr(self.Architecture.Structure[layer_input], 'output_'):
                raise Exception("The current layer ({layer_}) is referencing input layer {layer_input}, which does not have an output_ object defined.")
            
            ### If it's a residual connection, needs to be connected to its output Conv node
            if isinstance(layer_obj, Res):
                output_layer = self.Architecture.Structure[layer_obj.output_layer]
                layer_obj.connect_residual(output_layer)

            ### Getting the output_ of the input layer and passing it as an arg to forward pass
            input_ = self.Architecture.Structure[layer_input].get_output()
            layer_obj.forward_(input_)

            ### This is only for a softmax classification CNN right now.
            if layer_ == self.Architecture.reg["end"]:
                self.logits_ = layer_obj.logits_
                self.probs_ = layer_obj.probs_
    
        self.CEloss(Y_batch)

    def CEloss(self, Y_batch):
        self.loss = -np.sum(Y_batch * np.log(self.probs_)) ### Total batch loss
        self.batch_loss = -Y_batch * np.log(self.probs_) ### (64 x 1) loss

    def backwards_pass(self):
        
        end_layer = self.Architecture.reg.get("end")
        self.Architecture.Structure[end_layer].collect_backprop(self.batch_loss) ### Backprop for N samples in the batch

        for _, layer_obj in reversed(self.Architecture.Structure.items()):
            layer_input = layer_obj.layer_input
            input_layer_obj = self.Architecture.Structure[layer_input]
            layer_obj.backward_(input_layer_obj)

        for _, layer_obj in reversed(self.Architecture.Structure.items()):
            layer_obj.sgd_update()
    


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
                if layer.get("start") is not None:
                    print("Initializing the structure!")
                else:

                    print(f"No input layer assigned for layer {name}")
                    raise Exception("No input layer assigned.")
            
            ### If the layer input for the current layer does not exist, *and* this is not the start layer, raise an error                
            if Structure.get(layer["input"]) is None and layer.get("start") is None:
                key = "input"
                print(f"Layer {name} is trying to access layer {layer[key]}, which does not have an object assigned yet. Check order in the ordered Structure dict.")
                raise Exception("Incorrect input layer reference.")
            # if not hasattr(input_layer, 'k'):
            #     print(f"Layer {name} is trying to access the kernel shape from layer {layer[key]}, which does not have a kernel assigned yet. Check type assignment in the ordered Structure dict.")
            #     raise Exception("Incorrect input size assignment.")
            
        ### Generating layers
        ### I don't need to keep track of the image height/width, just the number of input kernels
            layer_obj = self.create_layer(name, layer, input_shape)
            if not isinstance(layer_obj, Res):
                input_shape = layer_obj.k
            Structure[name] = layer_obj

        return Structure

    def create_layer(self, name, layer, input_shape):
        layer_name = name
        layer_input = layer["input"]
        layer_type = layer["type"]
        params = layer["params"]

        ### In general, 'input shape' is just the number of kernels in the preceeding pool/conv layer.
        ### If the layer has a residual connection, this is handled during the residual forward pass, i.e. not by that layer itself.
        if layer_type == 'conv':
            layer_obj = Conv(name, params, layer_input, input_shape)

        if layer_type == 'pool':
            layer_obj = Pool(name, params, layer_input, input_shape)

        if layer_type == 'res':
            layer_obj = Res(name, params, layer_input, input_shape)
    
        if layer_type == 'fc':
            layer_obj = FC(name, params, layer_input, input_shape)
        
        if layer_type == 'adpool':
            layer_obj = Adaptive_Pool(name, params, layer_input, input_shape)
        
        return layer_obj


class Conv():
    def __init__(self, name, params, layer_input, input_shape):
        self.name = name
        self.layer_input = layer_input ### Previous layer that the input is derived from during forward pass
        self.input_shape = input_shape ### Number of kernels output from the previous layer
        self.k = params["k"] ### Number of kernels
        self.kernel_shape = params["shape"] ### Square dimension kernels assumed throughout, because why wouldn't I?
        self.stride = params["stride"] ### Stride
        self.activation = params["activation"]
        self.residual = False
        self.init_layer()

    def init_layer(self):
        self.init_weights()

    ### Using Kaiming initialization
    def init_weights(self):
        fan_in = self.kernel_shape * self.kernel_shape * self.input_shape
        stdev = np.sqrt(2.0/fan_in)
        ### 
        self.shape = (self.k, self.input_shape, self.kernel_shape, self.kernel_shape)
        self.weights = np.random.normal(0, stdev, self.shape)
        self.biases = np.zeros((1, 1, self.k)) ### cast against the matmul output with shape (X, H_out * W_out, k)

    ### For im2col, we flatten each kernel.
    def flatten(self):
        return self.weights.reshape(self.k, -1)

    ### I pad to the closest ceiling integer based on the equation at https://stats.stackexchange.com/questions/297678/how-to-calculate-optimal-zero-padding-for-convolutional-neural-networks
    def pad_input_(self):
        input_ = self.input_
        ### to account for floating point error, so the result is slightly below ceiling if an integer
        eps = 0.0001 
        P_h = int(np.ceil((((self.stride - 1) * input_.shape[2] - self.stride + self.kernel_shape) / 2) - eps))
        P_w = int(np.ceil((((self.stride - 1) * input_.shape[3] - self.stride + self.kernel_shape) / 2) - eps))
        ### Storing shape for backpropagation
        self.pad_shape = ((0,0) , (0, 0), (P_h, P_h), (P_w, P_w)) ### N (batch size) x F (filter number) x Padding columns x Padding rows
        return np.pad(input_, pad_width=self.pad_shape, mode = 'constant', constant_values = 0)

    ### Collecting the window for each output element, and only then flattening the window into columns.
    def image_to_column(self):
        input_ = self.padded_input_
        self.N, self.F_in, self.H_in, self.W_in = input_.shape
        kernel_shape = self.kernel_shape
        stride = self.stride

        ### Output shape        
        H_out = int((self.H_in - kernel_shape) // stride + 1)
        W_out = int((self.W_in - kernel_shape) // stride + 1)
        self.H_out = H_out
        self.W_out = W_out
        ### Here, a window is used to identify the input data necessary for each element in the output
        input_patches_ = np.zeros((self.N, self.F_in, kernel_shape, kernel_shape, H_out, W_out))

        ### Now we iterate over each element in the output, for each kernel, for each batch element.
        for i in range(kernel_shape):
            for j in range(kernel_shape):
                ### Iterate over N, Input Kernels, <ith window column>, <jth window row>
                imax = i + stride * H_out
                jmax = j + stride * W_out
                input_patches_[:, :, i, j, :, :] = input_[:, :, i:imax:stride, j:jmax:stride]

                ### Store in the respective flattened index for each N, Input Kernel, kernel column, kernel row, output column, output row

        ### Think of this as, for each N in the batch: (every kernel element) x (every output location)
        ### In other words, we're taking each window (for the corresponding output location), which can then just be 
        ### matrix multiplied with each kernel element. For F_in kernels of shape kernel_shape[0] x kernel_shape[1], 
        ### there's thus one element in this reshaped input.
        return input_patches_.transpose(0,4,5,1,2,3).reshape(self.N * self.H_out * self.W_out, self.F_in * self.kernel_shape * self.kernel_shape)

    ### Reshaping for the output :D
    def col_to_image(self):
        flattened_output = self.flattened_output
        return flattened_output.reshape(self.N, self.k, self.H_out, self.W_out)

### During forward pass, input is accessed within the method based on the "input" key and passed as an arg.
### As such, we don't need to return anything. Same goes for backprop - dL/d_ is passed backwards as an arg.
    def connect_residual_input(self, res_layer, residual_):
        self.residual = True
        self.res_layer = res_layer
        self.residual_ = residual_

    def forward_(self, input_):


        ### If a residual_ has been assigned by a residual connection, 
        if self.residual:

            ### Storing the layer_input for backprop, and updating the current vector X
            self.layer_input_ = input_
            if input_.shape[2] != self.residual_.shape[2] or input_.shape[3] != self.residual_.shape[3]:
                ### If the shapes don't match, we need to pad the residual (and store the padding)
                res_pad_H = input_.shape[2] - self.residual_.shape[2]
                res_pad_W = input_.shape[3] - self.residual_.shape[3]
                if res_pad_H % 2 != 0 or res_pad_W % 2 != 0:
                    raise Exception("padding is uneven? how did this even happen?")
                self.residual_padding = ((0,0), (0,0), (int(res_pad_H/2), int(res_pad_H/2)), (int(res_pad_W/2), int(res_pad_W/2)))
                self.residual_ = np.pad(self.residual_, pad_width = self.residual_padding, mode = 'constant', constant_values = 0)
            input_ += self.residual_
            self.input_ = input_
        else:
            self.input_ = input_
        ### Input channels known
        ### I'm storing all of these because all (or most) are needed for backpropagation.

        ### Padding the input
        self.N, _, self.input_h, self.input_w = input_.shape
        self.padded_input_ = self.pad_input_()
        ### Flattening the patches and flattening the kernels
        self.flattened_patches = self.image_to_column()
        self.flattened_weights = self.flatten()
        ### Convolution
        self.flattened_output = np.dot(self.flattened_patches, self.flattened_weights.T) + self.biases
        ### Unflattening
        self.pre_activation_output_ = self.col_to_image()
        self.output_ = activation_functions_(self.activation)(self.pre_activation_output_)
        self.backprop = np.zeros(self.output_.shape)

    def get_output(self):
        return self.output_
    
    ### Each layer has in its backward_ function, the input layer object that it is backpropagating towards
    def collect_backprop(self, backprop):
        self.backprop += backprop

    ### Backprop is the loss propagating backwards from later layers
    def backward_(self, input_layer_obj):
        
        ### dL/dF is just the convolution of X with backprop. Needs to be flattened, though.
        self.deactivated_backprop_ = activation_derivatives_(self.activation)(self.output_)
        self.flattened_backprop_ = self.deactivated_backprop_.reshape(self.flattened_output.shape)
        
        self.flattened_weight_update = np.dot(self.flattened_backprop_, self.flattened_weights) 
        
        ### Bias update
        self.bias_update = np.sum(self.backprop, axis = (0, 2, 3)) / (self.backprop.shape[0] * self.backprop.shape[2] * self.backprop.shape[3])

        ### Unraveling backprop
        self.dLdlayer_batch = self.flattened_backprop.reshape(self.N, self.input_shape, self.padded_input_.shape[2], self.padded_input_.shape[3])

        ### Remove padding (doesn't change anything if 0)
        H_unpad = self.padding[2]
        W_unpad = self.padding[3]
        H_end = self.dLdlayer_batch.shape[2] - H_unpad[1] if H_unpad[1] > 0 else None
        W_end = self.dLdlayer_batch.shape[3] - W_unpad[1] if W_unpad[1] > 0 else None
        self.dLdlayer_batch = self.dLdlayer_batch[:, :, H_unpad[0]:H_end, W_unpad[0]:W_end]        
        ### If there's a residual layer, backprop to it as well
        if self.residual:
            self.res_pad_H
            self.res_pad_W

            self.res_layer.collect_backprop(self.dLdlayer_batch)

#~~~~~~~#
        ### After calculations, backprop to the previous layer
        input_layer_obj.collect_backprop(self.dLdlayer_batch)

    def sgd_update(self, LR):
        self.weights -= LR * self.weights
        self.bias -= LR * self.bias_update
        return
    
class Pool():
    def __init__(self, name, params, layer_input, input_shape):
        self.name = name
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
    
    def get_output(self):
        return self.output_
    
    def forward_(self, input_):

        self.input_ = input_

        if self.global_:
            ### Flatten into (N x Nk x m * n) by condensing the last 2 axes
            self.flattened = input_.reshape(input_.shape[0], input_.shape[1], -1) 
            output_ = np.zeros((input_.shape[0], input_.shape[1]))

            if self.type_ == "AVG":
                ### The gradient is distributed as a fraction, i.e. propagated as
                ### (1 / ( kernel shape ^2 )) * dL/dz 
                ### for each node. This might be easier, since I can just store it at the time.
                self.output_ = np.mean(self.flattened, axis=2)
                self.backprop = np.zeros(self.output_.shape)
                return

            elif self.type_ == "MAX":
                ### Store indices of the 'max' values used as a boolean mask.
                ### Propagate backwards mask * dL/dz
                self.output_ = np.max(self.flattened, axis=2)
                self.backprop = np.zeros(self.output_.shape)
                return
        ### Kernel inputs is already known (or it should be)
        N, F_in, H_in, W_in = input_.shape
        assert F_in == self.input_shape
        H_out = (H_in - self.kernel_shape) // self.stride + 1
        W_out = (W_in - self.kernel_shape) // self.stride + 1
        self.output_ = np.zeros((N, F_in, H_out, W_out))
    
#### Might need to add padding here
        for i in range(H_out):
            for j in range(W_out):
                start_h = i * self.stride
                end_h = start_h + self.kernel_shape
                start_w = j * self.stride
                end_w = start_w + self.kernel_shape

                if self.type_ == "AVG":
                    self.output_[:,:, i, j] = np.mean(self.input_[:,:,start_h:end_h, start_w:end_w], axis = (2, 3))
                if self.type_ == "MAX":
                    self.output_[:,:, i, j] = np.max(self.input_[:,:,start_h:end_h, start_w:end_w], axis = (2, 3))

        self.backprop = np.zeros(self.output_.shape)


    def backward_(self, input_layer_obj):
        
        self.dLdlayer_batch = np.zeros(self.input_.shape)

        for i in range(self.backprop.shape[2]):
            for j in range(self.backprop.shape[3]):

                start_H = i * self.stride_H
                start_W = j * self.stride_W 
                end_H = start_H + self.stride_H
                end_W = start_W + self.stride_W
                
                if self.type_ == "AVG":
                    avg_size = self.H_prime * self.W_prime
                    self.dLdlayer_batch[:,:,start_H : end_H, start_W : end_W] = self.backprop[:,:,i,j] / avg_size
                
                if self.type_ == "MAX":

                    mask = (self.input_[:, :, start_H:end_H, start_W:end_W] == self.output_[:, :, i, j, np.newaxis, np.newaxis])
                    masked_backprop = np.zeros((self.input_.shape[0], self.input_.shape[1], end_H - start_H, end_W - start_W))
                    masked_backprop[mask] = 1

                    self.dLdlayer_batch[:, :, start_H:end_H, start_W:end_W] += masked_backprop * self.backprop[:, :, i, j, np.newaxis, np.newaxis]
        
        if self.padding_bool: 
            H_unpad = self.padding[2]
            W_unpad = self.padding[3]
            H_end = self.dLdlayer_batch.shape[2] - H_unpad[1] if H_unpad[1] > 0 else None
            W_end = self.dLdlayer_batch.shape[3] - W_unpad[1] if W_unpad[1] > 0 else None
            self.dLdlayer_batch = self.dLdlayer_batch[:, :, H_unpad[0]:H_end, W_unpad[0]:W_end]

        input_layer_obj.collect_backprop(self.dLdlayer_batch)

    def sgd_update(self, LR):
        return
    
### This is applied to each kernel. Number of kernels doesn't change. This is fine
class Adaptive_Pool():
    def __init__(self, name, params, layer_input, input_shape):
        self.name = name
        self.layer_input = layer_input
        self.input_shape = input_shape ### Input shape 
        self.k = params["k"] ### Output kernels
        self.output_shape = params["output_shape"] ### Flattened output shape (self.k * self.H_prime * self.W_prime) i.e. number of kernels in FC
        self.type_ = params["type"]
        self.global_ = params["global"]
        self.padding_bool = False
    
    ### Run this during execution.
    ### Might be slow, but right now it's only partly vectorized
    def forward_(self, input_):
        ### Input shape is of the form (N x Nk x m x n)
        ### Output shape is of the form (N x self.k * H_prime * W_prime)
        ### If global, it's easy - we just flatten and max/avg each row
        ### If local, it's not so easy - we run adaptive pooling, and only then flatten

        ### Only use global if the number of kernels in the final layer matches the 
        ### shape of the fully connected layer, defined in Structure["APn"]
        ### i.e. your last conv has 2048 kernels, and the FC layer has 2048 neurons
        if self.global_:

            ### Flatten into (N x Nk x m * n) by condensing the last 2 axes
            self.flattened = input_.reshape(input_.shape[0], input_.shape[1], -1) 
            self.input_ = input_
            

            if self.type_ == "AVG":
                ### The gradient is distributed as a fraction, i.e. propagated as
                ### (1 / ( kernel shape ^2 )) * dL/dz 
                ### for each node. This might be easier, since I can just store it at the time.
                self.output_ = np.mean(self.flattened, axis=2)
                self.backprop = np.zeros(self.output.shape)
                return

            elif self.type_ == "MAX":
                ### Store indices of the 'max' values used as a boolean mask.
                ### Propagate backwards mask * dL/dz
                self.output_ = np.max(self.flattened, axis=2)
                self.backprop = np.zeros(self.output.shape)
                return

        ### This is performed for each kernel. 
        self.input_ = input_
        N, Nk, H, W = input_.shape

        ### Assuming H_prime = W_prime, output shape is H_prime * W_prime * self.k = FC_neurons
        ### FC_Neurons = H_prime * 2 * self.k
        ### H_prime = np.sqrt( FC_neurons / self.k )
        H_prime = int(np.floor(np.sqrt(self.output_shape / self.k)))
        if (H_prime * H_prime * self.k) < self.output_shape:
            diff = (H_prime * H_prime * self.k) - self.output_shape
            H += diff
            W += diff
            if diff % 2 != 0:
                diff -= 1
                self.padding = ((0,0), (0,0), (1 + diff // 2, diff // 2), (1 + diff // 2, diff // 2))
                self.padded_input_ = np.pad(input_, pad_width = self.padding, mode = 'constant', constant_values=0)
            else:
                self.padding = ((0,0), (0,0), (diff // 2, diff // 2), (diff // 2, diff // 2))
                self.padded_input_ = np.pad(input_, pad_width = self.padding, mode = 'constant', constant_values=0)
            input_ = self.padded_input_
            self.padding_bool = True
            
        W_prime = H_prime

        stride_H = H // H_prime
        stride_W = W // W_prime

        #stride_H = 1
        #stride_W = 1

        self.stride_H = stride_H
        self.stride_W = stride_W
        self.H_prime = H_prime
        self.W_prime = W_prime

        self.output_ = np.zeros((N, self.k, H_prime, W_prime))

        for i in range(H_prime):
            for j in range(W_prime):
                start_H = i * stride_H
                start_W = j * stride_W 
                end_H = start_H + stride_H
                end_W = start_W + stride_W
                if self.type_ == "AVG":
                    self.output_[:, :, i, j] = np.mean(input_[:, :, start_H:end_H, start_W:end_W], axis = (2,3))
                    #self.output_.reshape(N, -1)

                elif self.type_ == "MAX":
                    self.output_[:, :, i, j] = np.max(input_[:, :, start_H:end_H, start_W:end_W], axis = (2,3))
                    #self.output_.reshape(N, -1)
        print(f"Output shape from the adaptive pooling layer is {self.output_.shape}")

        ### Can reshape during backprop via Nk, H_prime, W_prime to prepare for propagation backwards
        self.backprop = np.zeros(self.output_.shape)

    def get_output(self):
        return self.output_
    
    def collect_backprop(self, backprop):
        self.backprop += backprop

    def backward_(self, input_layer_obj):

        ### OH boy!

        self.dLdlayer_batch = np.zeros(self.input_.shape)

        for i in range(self.backprop.shape[2]):
            for j in range(self.backprop.shape[3]):
                start_H = i * self.stride_H
                start_W = j * self.stride_W 
                end_H = start_H + self.stride_H
                end_W = start_W + self.stride_W
                
                if self.type_ == "AVG":
                    avg_size = self.H_prime * self.W_prime
                    self.dLdlayer_batch[:,:,start_H : end_H, start_W : end_W] = self.backprop[:,:,i,j] / avg_size
                
                if self.type_ == "MAX":
                    ### (64 x 32 x 4 x 4)
                    mask = (self.input_[:, :, start_H:end_H, start_W:end_W] == self.output_[:, :, i, j, np.newaxis, np.newaxis])
                    masked_backprop = np.zeros((self.input_.shape[0], self.input_.shape[1], end_H - start_H, end_W - start_W))
                    masked_backprop[mask] = 1
                    #masked_backprop = np.zeros((self.input_.shape[0], self.input_.shape[1], end_H - start_H, end_W - start_W))
                    #masked_backprop = np.where(self.input_[:,:,start_H:end_H, start_W:end_W] == self.output_[:, :, i, j], 1, 0) 
                    ### this is so messy LMAO
                    self.dLdlayer_batch[:, :, start_H:end_H, start_W:end_W] += masked_backprop * self.backprop[:, :, i, j, np.newaxis, np.newaxis]
        if self.padding_bool: 
            H_unpad = self.padding[2]
            W_unpad = self.padding[3]
            self.dLdlayer_batch = self.dLdlayer_batch[:, :, H_unpad[0]: -H_unpad[1], W_unpad[0]: - W_unpad[1]]
        
        input_layer_obj.collect_backprop(self.dLdlayer_batch)
    def sgd_update(self, LR):
        return

### If there is a residual layer in a Conv() layer, this can be handled with the residual boolean
### i.e. enter the "if residual" statement and determine the backprop for both F(x) and x, and store
### Then during backward pass, corresponding input_layer can have its initialized (backprop) updated 
### before execution
class Res():
    def __init__(self, name, params, layer_input, input_shape):
        self.name = name
        self.layer_input = layer_input ### Previous layer that the input is derived from during forward pass
        self.input_shape = input_shape ### Number of kernels output from the previous layer
        self.output_layer = params["output"] ### Target output layer of the residual. During forward pass, output is added to this class.
        self.resize = params["resize"] ### If the shape should be resized. 

    def connect_residual(self, output_layer):
        self.output_layer_obj = output_layer

    ### I'm not resizing anything yet. If I implement this, this'll need to be updated.
    ### For now, with no resize, backprop shape is the same as input because the input is just fed forward
    def forward_(self, input_):
        self.output_layer_obj.connect_residual_input(self, input_)
        self.backprop = np.zeros(input_.shape)
    ### ^^^^ Haha, that was a funny joke. Turns out I'm just brute forcing it by resizing in the Conv layer anyways! 
    def get_output(self):
        return self.output_
    
    def collect_backprop(self, backprop):
        self.backprop += backprop

    def backward_pass(self, input_layer_obj):
        ### Just backprop * 1, no operation on the backpropagating loss
        input_layer_obj.collect_backprop(self.dL_dinput)

    def sgd_update(self, LR):
        return
    

class FC():
    def __init__(self, name, params, layer_input, input_shape):
        self.name = name
        self.layer_input = layer_input ### Input layer (usually a pooling layer)
        self.input_shape = input_shape ### Input shape (kernels)
        ### I'm using adaptive pooling. This needs to equal neurons
        ### If adaptive pooling isn't being used, you'll need to resize the input somehow else.
        self.labels = params["labels"] ### Number of classes
        self.k = params["shape"]    ### Ideal shape of the matrix, number of neurons
                                     ### In other words, for an input of shape [k * m * n],
                                     ### the matrix needs to be [k * m * n, Y] for Y labels
        if self.k != self.input_shape:
            ### This is fine, because I'm using kernel count. This should be 
            print(f"FC input shape is different from output shape - make sure to use adaptive pooling.")
        self.init_weight = params["init_weight"]
        self.activation = params["activation"]
        self.init_layer()

    def init_layer(self):
        ### Shape is e.g. 4096 x 10
        self.weights = np.random.randn(self.k, self.labels) * self.init_weight
        self.bias = np.zeros(self.labels) ### One bias per output channel
        return
    

    def get_output(self):
        return self.output_
    
    ### Should already be shaped correctly
    def forward_(self, input_):
        if input_.ndim != 2 and input_.shape[1] * input_.shape[2] * input_.shape[3] != self.k:
            raise Exception(f"Input_shape from final pooling layer is {input_.shape[1] * input_.shape[2] * input_.shape[3]}, whereas the neuron count is {self.k}. Check structure config or use adaptive pooling.")

        self.input_shape_ = input_.shape
        self.input_ = input_
        self.flattened_input_ = input_.reshape(input_.shape[0], -1)
        self.logits_ = np.dot(self.flattened_input_, self.weights) + self.bias
        self.probs_ = activation_functions_(self.activation)(self.logits_)

        self.output_ = self.probs_
        self.backprop = np.zeros(self.probs_.shape)

    def collect_backprop(self, backprop):
        print(f"backprop shape {backprop.shape}")
        self.backprop += backprop
    def backward_(self, input_layer_obj):
        ### Easy!
        ### f(x) = x * w + b

        ### dimensions are (self.shape * self.labels) * (self.labels) +  (self.labels) 
        ### so backprop dimensions just need to be self.labels to act on each
        self.deactivated_backprop_ = activation_derivatives_(self.activation)(self.output_)


        ### dL/dw
        self.weight_update = np.dot(self.flattened_input_.T, self.deactivated_backprop_)

        self.bias_update = np.sum(self.deactivated_backprop_, axis = 0)


        ### (N x 10) . (10 x 2048) -> (N x 2048), will need to be reshaped
        flattened_input_batch = np.dot(self.backprop, self.weights.T)
        self.dLdlayer_batch = flattened_input_batch.reshape(self.input_shape_)
        ### Backpropagating the full matrix of loss with N = 64 for dL/dX
        input_layer_obj.collect_backprop(self.dLdlayer_batch)

    def sgd_update(self, LR):
        self.weights -= LR * self.weight_update
        self.bias -= LR * self.bias_update




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
                # "shape":      kernel w x w, assume square kernels. (int)
                # "stride":     stride (int)
                # "activation": activation function, see activations.py

    ### Pooling layer params:
                # "k":          number of pooling kernels (int)
                # "stride":     stride (int)
                # "type":       whether 'AVG' or 'MAX' (string)
                # "global":     if global pooling (boolean)
                # "shape":      kernel w x w, assume square kernels. (int)
    
    ### Adaptive Pooling layer params:
                # "type":       whether 'AVG' or 'MAX' (string)
                # "global":     if global pooling (boolean)
                #               There is no kernel specification, because Nkin = Nkout.
                # "output_shape": Neurons in the FC layer that are being matched too

    ### Residual layer params:
                # "output":     name of the layer it's outputting too (string)
                # "resize":     whether the layer should be resizing to match the output layer (boolean)
                #               this could be dynamically determined as a function (if shape is wrong, 
                #               when called, reshape input to output)

    ### Fully connected layer params:
                # "labels":     should be input as a variable determining the shape 
                #               of the OH classification label. i.e. for MNISt, 10. (int)
                # "shape" :     number of neurons in the layer
                # "init_weight":scaling of the initialization for the layer
                # "activation": should be softmax but you can add others


# Ensure all names are unique!
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

####
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ OLD ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


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
    FC_neurons = hyperparams[4]

    Structure = OrderedDict([
        ("C1", {"type": 'conv', "input": None,
                "params": {"k": 32, "shape": 7, "stride": 2, "activation": 'relu'},
                "obj": None, "start": 1}),

        ("P1", {"type": 'pool', "input": "C1",
                "params": {"k": 32, "shape": 2, "stride": 1, "type": 'MAX', "global": 0},
                "obj": None}),

        ("R1", {"type": 'res', "input": "P1",
                "params": {"output": 'C4', "resize": 1},
                "obj": None}),

        ("C2", {"type": 'conv', "input": "P1",
                "params": {"k": 32, "shape": 2, "stride": 1, "activation": 'relu'},
                "obj": None}),

        ("C3", {"type": 'conv', "input": "C2",
                "params": {"k": 32, "shape": 2, "stride": 1, "activation": 'relu'},
                "obj": None}),

        ("C4", {"type": 'conv', "input": "C3",
                "params": {"k": 32, "shape": 2, "stride": 1, "activation": 'relu'},
                "obj": None}),

        ("C5", {"type": 'conv', "input": "C4",
                "params": {"k": 32, "shape": 2, "stride": 1, "activation": 'relu'},
                "obj": None}),

        ("C6", {"type": 'conv', "input": "C5",
                "params": {"k": 32, "shape": 2, "stride": 1, "activation": 'relu'},
                "obj": None}),

        ("C7", {"type": 'conv', "input": "C6",
                "params": {"k": 32, "shape": 2, "stride": 1, "activation": 'relu'},
                "obj": None}),

        ("AP1", {"type": 'adpool', "input": "C7",
                "params": {"type": 'MAX', "k": 32, "output_shape": FC_neurons, "global": 0},
                "obj": None}),

        ("FC", {"type": 'fc', "input": "AP1",
                "params": {"labels": labels.shape[1], "shape": FC_neurons, "init_weight": 0.01, "activation": 'softmax'}, #### Decide Y.shape elsewhere, placeholder
                "obj": None, "end": 1}),
        ])
    print(f"Shape of images are {features.shape}")
    img_channels = features.shape[1]
    Architecture = CNN_Architecture(Structure, img_channels)
    model = Model(Architecture)
    ### For each step, I split into train/validation sets.
    ### Pass validation sets into the class, train, 
    ### then test against validation, for each epoch

    for e in range(epochs):
    ### Split the validation set off here
        
        batch_count = 1
        for X_batch, Y_batch in gen_batches(features, labels, batch_size):
            batch_count+=1
            
            model.forward_pass(X_batch, Y_batch)

            loss = model.CEloss(Y_batch)

            print(f"Training loss is {loss} for batch {batch_count} out of {int(features.shape[0]/batch_size)}")

            model.backwards_pass()

    return model
            

