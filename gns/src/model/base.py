import torch
import torch.nn as nn


class FCNN(nn.Module):
    """Fully Connection Neural Network."""
    def __init__(self,
                 layers,
                 active_func,
                 norm=None
                 ):
        """
        Initialization FCNN.
            :param layers_list:
            :param active_func:
            :param norm: Function Point
        """
        super(FCNN, self).__init__()

        self.depth = len(layers) - 1
        self.layers = layers
        self.active_function = active_func

        if norm is not None:
            self.input_normalization = norm[0]
            self.output_normalization = norm[1]
        else:
            self.input_normalization = None
            self.output_normalization = None

        self.main = None

    def build(self):
        """Build the Model."""
        layers_list = list()
        for layer in range(self.depth - 1):
            layers_list.append(nn.Linear(self.layers[layer], self.layers[layer+1]))
            layers_list.append(self.active_function)
        layers_list.append(nn.Linear(self.layers[-2], self.layers[-1]))

        self.main = nn.Sequential(*layers_list)

    def init_network(self, init_method):
        """Initialize the param of network."""
        pass

    def init_network(self):
        """Initialize the param of network."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
                m.bias.data.zero_()

    def forward(self, x):
        assert self.main is not None, "Model Need to Be Built!!!"
        if self.input_normalization is not None:
            x = self.input_normalization(x)
        y = self.main(x)
        if self.output_normalization is not None:
            y = self.output_normalization(y)
        return y


if __name__ == '__main__':

    net_test = FCNN([1, 128, 128, 128, 2], active_func=nn.ReLU())
    net_test.build()
    print(net_test)
