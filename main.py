import torch
from torch import nn
from torch.optim import bui
from CNNLSTM import CNNLSTM
from utils import options
from utils.make_dataloader import get_loaders
from utils.scheduler import build_scheduler
import neptune.new as neptune
import os

def main(args):

    params = {
        "KERNEL_SIZES": args.kernel_sizes,
        "POOL_SIZES": args.pool_sizes,
        "CONV_CHANNELS": args.conv_channels,
        "NUM_LSTM_LAYERS": args.num_LSTM_layers,
        "LSTM_DIM": args.LSTM_dim,
        "BIDIRECTIONAL_LSTM": args.bidirectional_LSTM,
        "LSTM_DROPOUT": args.LSTM_dropout,
        "FC_DIMS": args.fc_dims,
        "FC_DROPOUT": args.fc_dropout,
        "BATCH_SIZE": args.batch_size,
        "ADAM_OPTIMIZER": args.adam_optimizer,
        "LEARNING_RATE": args.learning_rate,
        "SCHEDULER_USED": args.scheduler,
        "NUM_EPOCHS": args.num_epochs,
        "NUM_WORKERS": args.num_workers
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, test_loader, n_classes = get_loaders(batch_size= params['BATCH_SIZE'], 
                                                                   num_workers=params['NUM_WORKERS'], 
                                                                   path= os.path.join(args.dir, args.dataset_path), 
                                                                   return_whole_puzzle=False)
    
    kernel_sizes = [int(kernel_size) for kernel_size in args.kernel_sizez.split(',')]
    pool_sizes = [int(pool_size) for pool_size in args.pool_sizez.split(',')]
    conv_channels = [int(conv_channel) for conv_channel in args.conv_channels.split(',')]
    fc_dims = [int(fc_dim) for fc_dim in args.fc_dims.split(',')]

    model = CNNLSTM(input_size=28, 
                    num_classes=n_classes, 
                    in_channels=1, 
                    kernel_sizes=kernel_sizes,
                    pool_sizes=pool_sizes,
                    conv_channels=conv_channels,
                    num_LSTM_layers=params["NUM_LSTM_LAYERS"],
                    hidden_dim=params["LSTM_DIM"],
                    bidirectional_LSTM=params["BIDIRECTIONAL_LSTM"],
                    LSTM_dropout=params["LSTM_DROPOUT"],
                    fc_dims=fc_dims,
                    fc_dropout=params["FC_DROPOUT"]).to(device)


    loss_fn = nn.CrossEntropyLoss()
    if args.adam_optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr= args.learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr= args.learning_rate)

    if args.scheduler:
        scheduler = build_scheduler(optimizer, lr=params["LEARNING_RATE"])
    
    run = neptune.init_run(
            project=args.neptune_project,
            api_token=args.neptune_api_token
        )
    run["parameters"] = params

    def train(epoch):

        correct = 0
        total = 0
        correct_train = 0
        total_train = 0

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            x, _, sudoku_label = batch
            x, sudoku_label = x.to(device), sudoku_label.to(device)
            y = model(x)
            loss = loss_fn(y, sudoku_label)
            loss.backward()
            optimizer.step()
            predictions = torch.argmax(y, 1)
            total_train += x.shape[0]
            correct_train += predictions.eq(sudoku_label).sum().item()

            # Log training loss to Neptune
            run[f"train/loss"].log(loss.item())

            print(f'Epoch {epoch+1}, Loss: {loss.item():.2f}')
        train_accuracy = ((correct_train / total_train) * 100)
        print(f'Epoch {epoch+1}, Train accuracy: {train_accuracy:.2f}')
        run[f"train/accuracy"].log(train_accuracy)


        for batch_idx, batch in enumerate(val_loader):
            x, _, sudoku_label = batch
            x, sudoku_label = x.to(device), sudoku_label.to(device)
            y = model(x)
            predictions = torch.argmax(y, 1)
            total += x.shape[0]
            correct += predictions.eq(sudoku_label).sum().item()

        acc = (correct / total) * 100

        # Log validation accuracy to Neptune
        run[f"val/accuracy"].log(acc)

        print(f'Epoch {epoch+1}, acc: {acc:.2f}')
        if args.scheduler:
            scheduler.step()

    def test():
        model.eval()
        correct = 0
        total = 0
        for batch in test_loader:
            x, _, sudoku_label = batch
            x, sudoku_label = x.to(device), sudoku_label.to(device)
            y = model(x)
            predictions = torch.argmax(y, 1)
            total += x.shape[0]
            correct += predictions.eq(sudoku_label).sum().item()

        acc = (correct / total) * 100

        # Log testing accuracy to Neptune
        run[f"test/accuracy"].log(acc)

        print(f'Test accuracy: {acc:.2f}')

    for epoch in range(args.num_epochs):
        train(epoch)
    test()
    run.stop()



if __name__ == "__main__":
    args = options.read_command_line()
    main(args)