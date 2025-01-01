import itertools
import subprocess

# Define hyperparameter ranges


num_lstm_layers = [2, 3]
lstm_dim = [128, 256]
learning_rates = [0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001]
batch_sizes = [16, 64] # 64 instead of 128, training dataset contains 100 samples and with batch_size 128 returns none

# Generate all combinations
param_combinations = list(itertools.product(
    num_lstm_layers,
    lstm_dim,
    learning_rates,
    batch_sizes
))

# Fixed parameters
bidirectional_lstm = True
final_state = "" # False
fc_dims = 64
conv_channels = "32, 64"
kernel_sizes = "5, 5"
pool_sizes = "2, 2"
scheduler_used = "" # False
LSTM_dropout = 0.0
fc_dropout = 0.0
Adam_optimizer = True
num_epochs = 500
num_workers = 1

# Neptune credentials (update if necessary)
neptune_project = "GRAINS/visual-sudoku"
neptune_api_token = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly\
9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkMDQ2YThmNS1jNWU0LTQxZDItYTQxNy1lMGYzNTM4MmY5YTgifQ=="

# Loop over each combination
for params in param_combinations:

    lstm_layers, lstm_dim, lr, batch_size = params

    # Construct command
    command = f"""
    python ./main.py \
       --final_state "{final_state}" \
       --kernel_sizes "{kernel_sizes}" \
       --pool_sizes "{pool_sizes}" \
       --conv_channels "{conv_channels}" \
       --num_LSTM_layers "{lstm_layers}" \
       --LSTM_dim "{lstm_dim}" \
       --bidirectional_LSTM "{bidirectional_lstm}" \
       --LSTM_dropout "{LSTM_dropout}" \
       --fc_dims "{fc_dims}" \
       --fc_dropout "{fc_dropout}" \
       --batch_size "{batch_size}" \
       --adam_optimizer "{Adam_optimizer}" \
       --learning_rate "{lr}" \
       --scheduler "{scheduler_used}" \
       --num_epochs "{num_epochs}" \
       --num_workers "{num_workers}" \
       --neptune_project "{neptune_project}" \
       --neptune_api_token "{neptune_api_token}"
    """
    # command = f"""FINAL_STATE={}
    #              echo $FINAL_STATE"""

    # Execute the command
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # Print output in real-time
    for line in process.stdout:
        print(line.strip())

    # Print errors (if any)
    for line in process.stderr:
        print("ERROR:", line.strip())

    # Wait for the process to complete
    process.wait()
    print("Process completed with exit code:", process.returncode)
    # print("Executing:", command)
