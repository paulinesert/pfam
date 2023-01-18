import os 
import numpy as np
import torch 
import torch.nn as nn 


from argparse import ArgumentParser, Namespace
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.protCNN import ProtCNN
from utils.config import generate_config
from utils.data import BucketSampler, custom_collate_fn, PFAMDataset

def get_nb_trainable_parameters(parameters):
    # Compute the number of trainable parameters in a model 
    model_parameters = filter(lambda p: p.requires_grad, parameters)
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

def load_families_dict(run_folder):
    # Load the dict that maps the families to an integer 
    fam_dict_path = os.path.join(run_folder,'families_dict.pt')
    fam_dict = torch.load(fam_dict_path)
    return fam_dict


def test(
    model: nn.Module, 
    loss_fn: callable,
    dataloader: DataLoader
    ):
    """ Perform testing loop. 

    Args:
        model (nn.Module): the model that has already been trained
        loss_fn (callable): the loss used by the model during training
        dataloader (DataLoader): dataloader of the test set 

    Returns: 
            avg_loss (float): the loss averaged over all batches
            avg_accuracy (float): the averaged accuracy over the whole set 
    """

    running_n_samples = 0 
    running_correct_labels = 0 
    running_loss = 0 
    
    device = next(model.parameters()).device

    model.eval()

    # Loop over the batches 
    pbar = tqdm(dataloader)
    for batch_idx, batch in enumerate(pbar):
        inputs, targets, masks, _ = batch
        inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device)
        outputs = model(inputs, masks)
        
        loss = loss_fn(outputs, targets)

        # Make predictions
        predicted_class = torch.argmax(outputs, dim=1)
        labels = torch.argmax(targets, dim=1)

        # Compute averaged losses and other metrics 
        running_loss += loss.item()
        n_batches = batch_idx + 1 # because batch_idx starts at 0
        avg_loss = running_loss / n_batches 

        running_n_samples += inputs.shape[0]
        running_correct_labels += (predicted_class == labels).float().sum().item()
        avg_accuracy =  running_correct_labels / running_n_samples

    return avg_loss, avg_accuracy


def evaluate(
    data_hparams: Namespace, 
    model_hparams: Namespace, 
    train_hparams: Namespace, 
    model_dir: str
):
    """ Evaluate the model on the test dataset.
    data_hparams, model_hparams and train_hparams, respectively store the hyperparameters used by the data processing, the model and the training (here evaluation). 

    Args:
        data_hparams (Namespace): the hyperparameters for the data
        model_hparams (Namespace): the hyperparameters for the model
        train_hparams (Namespace): the hyperparameters for evaluating 
    """

    # Create test dataset
    families_dict = load_families_dict(model_dir)
    test_dataset = PFAMDataset('test', families_dict=families_dict, seq_lengths_bounds=data_hparams.seq_lengths_bounds, filter_fam=False, overwrite=data_hparams.overwrite)

    # Create samplers and dataloaders
    test_dl = DataLoader(test_dataset, collate_fn=custom_collate_fn, batch_size=train_hparams.batch_size, shuffle=False)
    
    # Create model 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_params_path = os.path.join(model_dir, 'model_weights.pt')
    n_classes = len(families_dict)
    model = ProtCNN(in_channels=model_hparams.in_channels, num_classes=n_classes, filters=model_hparams.filters, kernel_size=model_hparams.kernel_size, dilation_rate=model_hparams.dilation_rate, bottleneck_factor=model_hparams.bottleneck_factor, num_residual_block=model_hparams.num_residual_block)
    model.load_state_dict(torch.load(model_params_path, map_location='cpu'))
    model.to(device)
    print(f'The model has {get_nb_trainable_parameters(model.parameters())} trainable parameters.')

    # Create loss function 
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad(): 
        # Evaluation 
        avg_test_loss, avg_test_accuracy = test(model, loss_fn, test_dl)
    print(f'Test loss : {avg_test_loss}, accuracy: {avg_test_accuracy}')
        
if __name__ == '__main__': 
    parser = ArgumentParser(description="Config file")
    parser.add_argument(
        "--config_file_path",
        help="path to the config file",
        default="./config/baseline_config.yaml",
    )
    parser.add_argument(
        "--logdir",
        help="Folder that stores the model weights and families dict", 
    )    
    args = parser.parse_args()
    config = generate_config(args.config_file_path) # turn YAML config file into nested Namespace args
    print(config)
    evaluate(config.data, config.model, config.train, args.logdir)

