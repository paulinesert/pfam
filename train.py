import os 
import numpy as np
import torch 
import torch.nn as nn 


from argparse import ArgumentParser, Namespace
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from time import time_ns
from tqdm import tqdm

from models.protCNN import ProtCNN
from utils.config import generate_config
from utils.data import BucketSampler, custom_collate_fn, PFAMDataset

def get_nb_trainable_parameters(parameters):
    # Compute the number of trainable parameters in a model 
    model_parameters = filter(lambda p: p.requires_grad, parameters)
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

def get_train_eval_loop(
    model: nn.Module,
    optimizer: torch.optim.Optimizer, 
    loss_fn: callable, 
    writer, 
    ) -> callable:
    """ Generate the function that performs a training/validation loop over the given partition (train or validation).

    Args:
        model (nn.Module): the model
        optimizer (torch.optim.Optimizer): the optimizer
        loss_fn (callable): the loss function
        writer (_type_): the writer (Tensorboard) to store the logs

    Returns:
        callable: the function that performs the training / validation loop 
    """

    global global_train_step, global_val_step
    global_train_step = 0 
    global_val_step = 0 

    def train_eval_loop(
        is_train: bool, 
        dataloader: DataLoader, 
        epoch: int, 
        log_step: int, 
    ) -> tuple: 
        """ Compute one epoch (training) or compute loss and metrics over the validation set.

        Args:
            is_train (bool): If True, the model is in training mode, it is eval mode otherwise.
            dataloader (DataLoader): the partition's dataloader
            epoch (int): the current epoch
            log_step (int): Each log_step train/val steps the metrics are logged. If None, no logging is performed 
        """
        # 
        global global_train_step, global_val_step
        
        running_n_samples = 0 
        running_correct_labels = 0 
        running_loss = 0 
        

        device = next(model.parameters()).device

        if is_train: 
            model.train()
        else:
            model.eval()

        # Loop over the batches 
        pbar = tqdm(dataloader)
        for batch_idx, batch in enumerate(pbar):
            inputs, targets, masks, _ = batch
            inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device)
            outputs = model(inputs, masks)

            if is_train: 
                optimizer.zero_grad()
            
            loss = loss_fn(outputs, targets)

            if is_train: 
                loss.backward()
                optimizer.step()

            # Make predictions
            predicted_class = torch.argmax(outputs, dim=1)
            labels = torch.argmax(targets, dim=1)
            batch_accuracy = (predicted_class == labels).float().mean().item()

            # Compute averaged losses and other metrics 
            running_loss += loss.item()
            n_batches = batch_idx + 1 # because batch_idx starts at 0
            avg_loss = running_loss / n_batches 

            running_n_samples += inputs.shape[0]
            running_correct_labels += (predicted_class == labels).float().sum().item()
            avg_accuracy =  running_correct_labels / running_n_samples

            # Log info and metrics 
            if is_train: 
                global_train_step += 1
                if log_step and (global_train_step % log_step) == 0: 
                    lr = optimizer.param_groups[0]['lr']
                    writer.add_scalar('train/loss', loss.item(), global_train_step)
                    writer.add_scalar('train/accuracy', batch_accuracy, global_train_step)
                    writer.add_scalar('train/learning_rate', lr, global_train_step)
                    pbar.set_description(f'Train : Epoch {epoch} - loss: {avg_loss:.2f} - accuracy: {avg_accuracy:.2f}')



            else: 
                global_val_step += 1
                if log_step and (global_val_step % log_step) == 0 : 
                    writer.add_scalar('val/loss', loss.item(), global_val_step)
                    writer.add_scalar('val/accuracy', batch_accuracy, global_val_step)
                    pbar.set_description(f'Val : Epoch {epoch} - loss: {avg_loss:.2f} - accuracy: {avg_accuracy:.2f}')

        return avg_loss, avg_accuracy

    return train_eval_loop


def run(
    data_hparams: Namespace, 
    model_hparams: Namespace, 
    train_hparams: Namespace,
    test: bool, 
    store: bool
):
    """ Perform training and evaluation steps for a given number of epochs.
    data_hparams, model_hparams and train_hparams, respectively store the hyperparameters used by the data processing, the model and the training. 

    Args:
        data_hparams (Namespace): the hyperparameters for the data
        model_hparams (Namespace): the hyperparameters for the model
        train_hparams (Namespace): the hyperparameters for the training
        test (bool): If True, the test set is also evaluated
        store (bool): If True, the weights of the model and the dict of families are stored
    """

    # Create datasets for all the train and val partitions
    train_dataset = PFAMDataset('train', families_dict=None, seq_lengths_bounds=data_hparams.seq_lengths_bounds, filter_fam=data_hparams.filter_fam, n_fam=data_hparams.n_fam, overwrite=data_hparams.overwrite)
    val_dataset = PFAMDataset('dev', families_dict=train_dataset.families_dict, seq_lengths_bounds=data_hparams.seq_lengths_bounds, filter_fam=False, overwrite=data_hparams.overwrite)
    
    # Create samplers and dataloaders
    train_dl =  DataLoader(train_dataset, batch_size=train_hparams.batch_size,  collate_fn=custom_collate_fn, shuffle=False)
    val_dl = DataLoader(val_dataset, collate_fn=custom_collate_fn, batch_size=train_hparams.batch_size, shuffle=False)
    
    # Create model 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = len(train_dataset.families_dict)
    model = ProtCNN(in_channels=model_hparams.in_channels, num_classes=n_classes, filters=model_hparams.filters, kernel_size=model_hparams.kernel_size, dilation_rate=model_hparams.dilation_rate, bottleneck_factor=model_hparams.bottleneck_factor, num_residual_block=model_hparams.num_residual_block)
    model.to(device)
    print(f'The model has {get_nb_trainable_parameters(model.parameters())} trainable parameters.')


    # Create loss function 
    loss_fn = nn.CrossEntropyLoss()

    # Create optimizer 
    optimizer = torch.optim.Adam(model.parameters(), lr=train_hparams.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, factor=train_hparams.lr_factor, patience=train_hparams.lr_patience)

    # Create logger (Tensorboard)
    log_filepath = os.path.join(train_hparams.log_dir, str(time_ns()))
    writer = SummaryWriter(log_dir=log_filepath)
    print(f'Logs will be saved at {log_filepath}')

    # Get train / eval loop function 
    train_eval_loop = get_train_eval_loop(model, optimizer, loss_fn, writer)

    # Training / eval loop : 
    val_loss = float('inf')
    prev_loss = float('inf')
    dec = 0

    for epoch in range(train_hparams.n_epochs):
        # Training
        _, _ = train_eval_loop(is_train=True, dataloader=train_dl, epoch=epoch, log_step=train_hparams.log_step)

        with torch.no_grad(): 
            # Validation 
            val_loss, _ = train_eval_loop(is_train=False, dataloader=val_dl, epoch=epoch, log_step=train_hparams.log_step)
            scheduler.step(val_loss)

        # Early stopping 
        if val_loss > prev_loss : 
            dec += 1 
        else:
            dec = 0 
    
        if dec == train_hparams.lr_patience + 2: 
            break 
        prev_loss = val_loss


    if test: 
        # Evaluation 
        test_dataset = PFAMDataset('test', families_dict=train_dataset.families_dict, seq_lengths_bounds=data_hparams.seq_lengths_bounds, filter_fam=False, overwrite=data_hparams.overwrite)
        test_dl = DataLoader(test_dataset, collate_fn=custom_collate_fn, batch_size=train_hparams.batch_size, shuffle=False)
        avg_test_loss, avg_test_acc = train_eval_loop(is_train=False, dataloader=test_dl, epoch=epoch, log_step=None) 
        print(f'Test : Epoch {epoch} - loss: {avg_test_loss:.2f} - accuracy: {avg_test_acc:.2f}')

    if store:
        torch.save(model.state_dict(), os.path.join(train_hparams.log_dir,'model_weights.pt'))
        torch.save(train_dataset.families_dict, os.path.join(train_hparams.log_dir,'families_dict.pt'))
        
if __name__ == '__main__': 
    parser = ArgumentParser(description="Config file")
    parser.add_argument(
        "--config_file_path",
        help="Path to the config file",
        default="./config/baseline_config.yaml",
    )
    parser.add_argument(
        "--test",
        action='store_true',
        help="If True, the test set is also evaluated. Default is False."
    )
    parser.add_argument(
        "--store",
        action='store_false',
        help="If True, the weights of the model are stored. Default is True.",
    )
    args = parser.parse_args()
    config = generate_config(args.config_file_path) # turn YAML config file into nested Namespace args
    print(config)
    os.makedirs(config.train.log_dir, exist_ok=True) # create the dir where the logs will be stored
    run(config.data, config.model, config.train, args.test, args.store)
