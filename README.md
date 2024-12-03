# VisualSudoku

## Installation

1. Clone this repository to your machine:
   ```bash
   git clone https://github.com/Amir-Goudarzi/CNN-LSTM
2. Navigate to the project directory:
   ```bash
   cd CNN-LSTM
3. Install the required Python packages from requirements.txt:
   ```bash
   pip install -r requirements.txt

## Usage and Configuration 

   Use the following script to specify the parameters and train the model. 

   ```bash
   KERNEL_SIZES="5, 5"
   POOL_SIZES="2, 2"
   CONV_CHANNELS="32, 64"
   NUM_LSTM_LAYERS=3
   LSTM_DIM=128
   BIDIRECTIONAL_LSTM=False
   LSTM_DROPOUT=0.0
   FC_DIMS="64"
   FC_DROPOUT=0.0
   BATCH_SIZE=10
   ADAM_OPTIMIZER=True
   LEARNING_RATE=0.0001
   SCHEDULER_USED=False
   NUM_EPOCHS=10
   NUM_WORKERS=1
   
   python ./main.py \
      --kernel_sizes "$KERNEL_SIZES" \
      --pool_sizes "$POOL_SIZES" \
      --conv_channels "$CONV_CHANNELS" \
      --num_LSTM_layers "$NUM_LSTM_LAYERS" \
      --LSTM_dim "$LSTM_DIM" \
      --bidirectional_LSTM "$BIDIRECTIONAL_LSTM" \
      --LSTM_dropout "$LSTM_DROPOUT" \
      --fc_dims "$FC_DIMS" \
      --fc_dropout "$FC_DROPOUT" \
      --batch_size "$BATCH_SIZE" \
      --adam_optimizer "$ADAM_OPTIMIZER" \
      --learning_rate "$LEARNING_RATE" \
      --scheduler "$SCHEDULER_USED" \
      --num_epochs "$NUM_EPOCHS" \
      --num_workers "$NUM_WORKERS"
