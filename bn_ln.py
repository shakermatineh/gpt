import numpy as np
import torch
import torch.nn as nn


class BatchNorm:
    # Define a class to keep state of running stats
    def __init__(self, num_features, momentum=0.9, eps=1e-5):
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.running_mean = np.zeros((num_features,))
        self.running_var = np.ones((num_features,))
        self.gamma = np.ones((num_features,))
        self.beta = np.zeros((num_features,))
    
    def forward(self, x, training=True):
        if training:
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
            
            x_normalized = (x - batch_mean) / np.sqrt(batch_var + self.eps)
        else:
            x_normalized = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
        
        # Apply learned scale and shift.
        out = self.gamma * x_normalized + self.beta
        return out

# Batchnorm using numpy
batch_size, num_features = 10, 5
batchnorm = BatchNorm(num_features, momentum=0.9)
x = np.random.randn(batch_size, num_features)
print(f"Output (training): {batchnorm.forward(x, training=True)}")
print(f"Output (inference): {batchnorm.forward(x, training=False)}")


def layer_norm_forward(x, gamma, beta, eps=1e-5):
    mu = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    
    # Normalize
    x_normalized = (x - mu) / np.sqrt(var + eps)
    
    # Apply learned scale (gamma) and shift (beta)
    out = gamma * x_normalized + beta
    return out

# Example inputs
x = np.random.randn(10, 5)
gamma = np.ones((5,))
beta = np.zeros((5,))       
output = layer_norm_forward(x, gamma, beta)
print(output.shape)  # Output shape remains the same as input shape


# Batchnorm using pytorch
batch_size, num_features = 10, 5
x = torch.randn(batch_size, num_features)
batchnorm = nn.BatchNorm1d(num_features=num_features, momentum=0.9)
batchnorm.train()
print(f"Output Pytorch (training): {batchnorm(x)}")
batchnorm.eval()
print(f"Output Pytorch (inference): {batchnorm(x)}") # Using running stats at Inference

# Layernorm using pytorch
layernorm = nn.LayerNorm(num_features) # normalizes across features
print(f"Layernor output Pytorch: {layernorm(x)}")  # works same at training and inference.