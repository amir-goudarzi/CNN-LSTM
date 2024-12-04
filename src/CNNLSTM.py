from torch import nn
import math

def getPair(pair, num):
    if isinstance(pair, tuple):
        return pair
    else:
        return tuple(pair for _ in range(num))
    

def calculate_output_size(input_size=28, kernel_size=5, pool_size=2, padding=2, stride=1, pool_stride=None):
    if pool_stride is None:
        pool_stride = pool_size 
    conv_output = math.floor(((input_size + (2 * padding) - kernel_size) / stride) + 1)
    pool_output = math.floor((conv_output - pool_size) / pool_stride + 1)
    return pool_output


def create_conv_layer(in_channels, out_channels, kernel_size=5, pool_size=2, padding=2, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=pool_size)
    )

def create_cnn(input_size=28, in_channels=1, conv_channels=[32, 64], kernel_sizes=[5, 5], pool_sizes=[2, 2], paddings=[1, 1], strides=[2, 2]):
    layers = []
    current_size = input_size
    for _, (out_channels, kernel_size, pool_size, padding, stride) in enumerate(
        zip(conv_channels, kernel_sizes, pool_sizes, paddings, strides)):
        
        layers.append(create_conv_layer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            pool_size=pool_size,
            padding=padding,
            stride=stride
        ))

        current_size = calculate_output_size(current_size, kernel_size, pool_size=pool_size, padding=padding, stride=stride)
        
        in_channels = out_channels  # Update in_channels for the next layer

    feature_size = current_size * current_size * conv_channels[-1]

    return nn.Sequential(*layers), feature_size


def create_fc_layer(input_dim, output_dim, dropout=0.0):
    return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),  
    )


def create_fc(input_dim=128, layer_dims=[64], num_classes=2, dropout=0.0):
    layers = []
    dims = [input_dim]
    dims.extend(layer_dims)

    for i in range(len(dims)-1):
        layers.append(create_fc_layer(input_dim=dims[i], output_dim=dims[i+1], dropout=dropout))

    layers.append(nn.Linear(in_features=dims[-1], out_features=num_classes))

    return nn.Sequential(*layers)



class CNNLSTM(nn.Module):
    def __init__(self, final_state=False, input_size=28, num_patches=16, num_classes=2, in_channels= 1, kernel_sizes=[5, 5], pool_sizes=[2, 2], 
                 conv_channels=[32, 64], num_LSTM_layers=3, hidden_dim=128, bidirectional_LSTM=False, LSTM_dropout=0.0, 
                 fc_dims=[64], fc_dropout=0.0):
        """
        :param num_patches: Number of patches in the input (default: 16 for 4x4 Sudoku).
        :param num_classes: Number of output classes (default: 10 for digits 0-9).
        :param feature_dim: Output feature dimension from CNN.
        :param hidden_dim: Number of hidden units in LSTM.
        """
        super().__init__()
        self.final_state = final_state 

        self.cnn, self.cnn_output_dim = create_cnn(input_size=input_size, in_channels=in_channels, conv_channels=conv_channels, 
                                                   kernel_sizes=kernel_sizes, pool_sizes=pool_sizes)   
    
        self.lstm = nn.LSTM(input_size=self.cnn_output_dim, hidden_size=hidden_dim, num_layers=num_LSTM_layers, batch_first=True,
                            bidirectional=bidirectional_LSTM, dropout=LSTM_dropout)
        
        if final_state:
            lstm_output_dim = hidden_dim 
        if not final_state:
            lstm_output_dim = hidden_dim * num_patches * (2 if bidirectional_LSTM else 1) 

        self.fc = create_fc(input_dim=lstm_output_dim, layer_dims=fc_dims, num_classes=num_classes, dropout=fc_dropout)
        
        

    def forward(self, x):
        """
        :param x: Input tensor of shape [batch_size, num_patches, 28, 28].
        :return: Output tensor of shape [batch_size, num_patches, num_classes].
        """
        batch_size, num_patches, height, width = x.shape
        
        # Reshape input to process each patch with the CNN
        x = x.view(batch_size * num_patches, 1, height, width)  # Combine batch and patch dimensions
        
        # Extract features using the CNN
        features = self.cnn(x)  # Shape: [batch_size * num_patches, cnn_output_dim]
        
        # Reshape features back into a sequence for the LSTM
        features = features.view(batch_size, num_patches, -1)  # Shape: [batch_size, num_patches, cnn_output_dim]
        
        # Process the sequence with the LSTM
        lstm_out, (h_n, c_n) = self.lstm(features)  # Shape: [batch_size, num_patches, hidden_dim]
        
        if self.final_state:
            fc_input = h_n[-1]
        else:
            # Flatten 
            fc_input = lstm_out.reshape(batch_size, -1)

        # Apply the fully connected layer to each patch output
        outputs = self.fc(fc_input)  # Shape: [batch_size, num_patches, num_classes]
        
        return outputs
