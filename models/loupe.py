import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PoolingBaseModel(nn.Module):
    """Inherit from this class when implementing new models."""

    def __init__(self, feature_size, max_samples, cluster_size, output_dim,
                 gating=True, add_batch_norm=True, is_training=True):
        super(PoolingBaseModel, self).__init__()

        self.feature_size = feature_size
        self.max_samples = max_samples
        self.output_dim = output_dim
        self.is_training = is_training
        self.gating = gating
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size

    def forward(self, reshaped_input):
        raise NotImplementedError("Models should implement the forward pass.")

    def context_gating(self, input_layer):
        """Context Gating

        Args:
        input_layer: Input layer in the following shape:
        'batch_size' x 'number_of_activation'

        Returns:
        activation: gated layer in the following shape:
        'batch_size' x 'number_of_activation'
        """


        input_dim = input_layer.size(1)

        gating_weights = nn.Parameter(torch.randn(input_dim, input_dim) * (1.0 / math.sqrt(input_dim))).cuda()
        gates = torch.matmul(input_layer, gating_weights)

        if self.add_batch_norm:
            gates = nn.BatchNorm1d(input_dim).cuda()(gates)
        else:
            gating_biases = nn.Parameter(torch.randn(input_dim) * (1.0 / math.sqrt(input_dim))).cuda()
            gates += gating_biases

        gates = torch.sigmoid(gates)
        activation = input_layer * gates

        return activation


class NetVLAD(PoolingBaseModel):
    """Creates a NetVLAD class."""
    def __init__(self, feature_size = 768, max_samples=1, cluster_size=30, output_dim=384,
                 gating=True, add_batch_norm=True, is_training=True):
        super(NetVLAD, self).__init__(
            feature_size=feature_size,
            max_samples=max_samples,
            cluster_size=cluster_size,
            output_dim=output_dim,
            gating=gating,
            add_batch_norm=add_batch_norm,
            is_training=is_training)

        self.cluster_weights = nn.Parameter(torch.randn(feature_size, cluster_size) * (1.0 / math.sqrt(feature_size)))
        self.hidden1_weights = nn.Parameter(torch.randn(cluster_size * feature_size, output_dim) * (1.0 / math.sqrt(cluster_size)))

    def forward(self, reshaped_input):
        """Forward pass of a NetVLAD block.

        Args:
        reshaped_input: Input shape: 'batch_size' x 'max_samples' x 'feature_size'

        Returns:
        vlad: the pooled vector of size: 'batch_size' x 'output_dim'
        """
        batch_size = reshaped_input.size(0)

        activation = torch.matmul(reshaped_input, self.cluster_weights)
        if self.add_batch_norm:
            activation = nn.BatchNorm1d(self.cluster_size).cuda()(activation.view(-1, self.cluster_size)).view(batch_size, self.max_samples, self.cluster_size)
        else:
            cluster_biases = nn.Parameter(torch.randn(self.cluster_size) * (1.0 / math.sqrt(self.feature_size))).cuda()
            activation += cluster_biases

        activation = F.softmax(activation, dim=-1)
        activation = activation.view(batch_size, self.max_samples, self.cluster_size)

        a_sum = torch.sum(activation, dim=-2, keepdim=True)
        cluster_weights2 = nn.Parameter(torch.randn(1, self.feature_size, self.cluster_size) * (1.0 / math.sqrt(self.feature_size))).cuda()
        a = a_sum * cluster_weights2

        activation = activation.permute(0, 2, 1)
        reshaped_input = reshaped_input.view(batch_size, self.max_samples, self.feature_size)

        vlad = torch.matmul(activation, reshaped_input)
        vlad = vlad.permute(0, 2, 1) - a

        vlad = F.normalize(vlad, p=2, dim=1)
        vlad = vlad.reshape(batch_size, -1)
        vlad = F.normalize(vlad, p=2, dim=1)

        vlad = torch.matmul(vlad, self.hidden1_weights)

        if self.gating:
            vlad = self.context_gating(vlad)
        return vlad
    
if __name__=='__main__':
    feature_size = 768
    max_samples = 1
    cluster_size = 10
    output_dim = 768
    gating = True
    add_batch_norm = True
    is_training = True
    model = NetVLAD(feature_size, max_samples, cluster_size, output_dim, gating, add_batch_norm, is_training)
    reshaped_input = torch.randn(2, 1, 768)
    output = model(reshaped_input)
    # print(output.shape)