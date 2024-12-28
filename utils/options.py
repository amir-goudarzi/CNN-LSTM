import argparse

def read_command_line():

    parser = argparse.ArgumentParser(description='CNN+LSTM')

    # Model parameters
    parser.add_argument('--final_state', type=bool, required=False, default=False, help='Uses only the final hidden state')
    parser.add_argument('--kernel_sizes', type=str, required=False, default="5, 5", help='Kernel sizes for each conv layer')
    parser.add_argument('--pool_sizes', type=str, required=False, default="2, 2", help='Pool sizes for each conv layer')
    parser.add_argument('--conv_channels', type=str, required=False, default="32, 64", help='Output channels for each conv layer')
    parser.add_argument('--num_LSTM_layers', type=int, required=False, default=3, help='number of LSTM layers')
    parser.add_argument('--LSTM_dim', type=int, required=False, default=128, help='Embedding dimenstion for LSTM')
    parser.add_argument('--bidirectional_LSTM', type=bool, required=False, default=False, help='True for bidirectional LSTM')
    parser.add_argument('--LSTM_dropout', type=float, required=False, default=0.0, help='Probability of dropout in LSTM')
    parser.add_argument('--fc_dims', type=str, required=False, default="64", help='Dimensions of each fully connected layer')
    parser.add_argument('--fc_dropout', type=float, required=False, default=0.0, help='Probability of dropout in fully connected network')

    # Training parameters
    parser.add_argument('--batch_size', type= int, required=False, default=10, help='Batch size')
    parser.add_argument('--adam_optimizer', type= bool, required=False, default=True, help='True for Adam, False for SGD')
    parser.add_argument('--learning_rate', type= float, required=False, default=0.0001, help='Learning rate')
    parser.add_argument('--scheduler', type= bool, required=False, default=False, help='True for using a learning rate scheduler')
    parser.add_argument('--num_epochs', type= int, required=False, default=10, help='Number of epochs')
    parser.add_argument('--num_workers', type= int, required=False, default=1, help='Number of workers')

    # Other
    parser.add_argument('--dir', type=str, default='/content/drive/MyDrive/Colab Notebooks/',
                    help='Data directory')
    parser.add_argument('--dataset_path', default='VisualSudoku', type=str, help='The path to the training data.')
    parser.add_argument("--neptune_project", type=str, help="Neptune project directory")
    parser.add_argument("--neptune_api_token", type=str, help="Neptune api token")
    parser.add_argument("--split", type=int, required=False, default=10, help="Data split")

    args = parser.parse_args()
    return args