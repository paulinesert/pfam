import numpy as np 
import os 
import pandas as pd 
import random
import re
import time
import torch 
import torch.nn.functional as F

from collections import defaultdict, Counter
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, default_collate


DATA_DIR_PATH = os.path.join(os.path.dirname(__file__), os.pardir, 'data','random_split')

RGX_PATTERN_UNWANTED_AA = '[XUBOZ]+'

DICT_AA = {
 'L': 0,
 'A': 1,
 'V': 2,
 'G': 3,
 'E': 4,
 'S': 5,
 'I': 6,
 'R': 7,
 'D': 8,
 'K': 9,
 'T': 10,
 'P': 11,
 'F': 12,
 'N': 13,
 'Q': 14,
 'Y': 15,
 'M': 16,
 'H': 17,
 'C': 18,
 'W': 19
}

def AA_to_int(seq: str) -> torch.Tensor:
    # Remove uncommon AA 
    seq_common = re.sub(RGX_PATTERN_UNWANTED_AA, '', seq)
    seq_processed = torch.as_tensor([DICT_AA[AA] for AA in seq_common])
    return seq_processed

def fam_to_int(fam: str, families_dict: dict) -> torch.Tensor:
    # Map a family accession to an int 
    fam_processed = torch.as_tensor(families_dict[fam])
    return fam_processed

def int_to_onehot(seq: str, nbr_classes: int) -> torch.Tensor:
    # Convert int encoding to one hot encoding
    return F.one_hot(seq, num_classes=nbr_classes)
    
def seq_to_onehot(seq: str) -> torch.Tensor: 
    # One-hot encode a sequence of amino acids 
    int_encoding = AA_to_int(seq)
    onehot_encoding = int_to_onehot(int_encoding, nbr_classes=len(DICT_AA))
    return onehot_encoding

def family_to_onehot(fam: str, families_dict: dict) -> torch.Tensor:
    # One-hot encode a family succession
    int_encoding = fam_to_int(fam, families_dict)
    onehot_encoding = int_to_onehot(int_encoding, nbr_classes=len(families_dict))
    return onehot_encoding

def generate_family_int_mapping(families: list) -> dict:
    # Generate the dict that maps a family accession to an integer
    index =   list(map(int, range(0,len(families))))
    families_mapping = dict(zip(families, index))
    return families_mapping 

def compute_most_freq_fam(families: list, n_fam: int) -> list: 
    # Computes the list of the n_fam most frequent families in the training set 
    counter = Counter(families).most_common()
    most_freq_fam = list(dict(counter[0:n_fam]).keys())
    return most_freq_fam

def read_all_shards(partition: str, data_dir: str) -> pd.DataFrame:
    """Read the shared data files.
    Args:
        partition (str): train, dev or test 
        data_dir (str): directory where the data is stored 
    
    Returns:
        pd.DataFrame: partition data 
    """
    shards = []
    for fn in os.listdir(os.path.join(data_dir, partition)):
        with open(os.path.join(data_dir, partition, fn)) as f:
            shards.append(pd.read_csv(f, index_col=None))
    df = pd.concat(shards) 
    df['seq_len'] = df['sequence'].apply(len)
    return df



class PFAMDataset(Dataset):
    def __init__(self,
            partition: str,
            families_dict: dict,
            seq_lengths_bounds: tuple = (None, None),  
            filter_fam: bool = True, 
            n_fam: int = 100, 
            overwrite: bool = False, 
            ):
        """PFAM Dataset. 

        Load / preprocess the given partition. 

        Args:
            partition (str): 'train', 'dev' or 'test
            families_dict (dict): If provided, a dict mapping each family accession to an integer
            seq_lengths_bounds (tuple, optional): If provided, the bounds (b_min, b_max) in which the sequences' lengths must be,
                 if not no filtering on the seq lengths is applied. Defaults to (None, None).
            filter_fam (bool, optional): If True, then the dataset is filtered to only keep the n_fam most frequent families in the training set. Defaults to True.
            n_fam (int, optional): Number of most frequent families in the training set to keep. Defaults to 500.
            overwrite (bool, optional): If True, preprocess the dataset even if an already processed version exists. Defaults to False.
        """
        super(PFAMDataset).__init__()
        self.partition = partition 
        self.families_dict = families_dict 
        self.seq_lengths_bounds = seq_lengths_bounds
        self.filter_fam = filter_fam
        self.n_fam = n_fam
        self.raw_dir_path = DATA_DIR_PATH
        if self.seq_lengths_bounds != (None, None): 
            self.processed_dir_path = os.path.join(DATA_DIR_PATH + '_processed_' + '-'.join(seq_lengths_bounds), partition)
        else: 
            self.processed_dir_path = os.path.join(DATA_DIR_PATH + '_processed', partition)

        self.processed_file_path = os.path.join(self.processed_dir_path, partition+'.pt')

        if (not os.path.exists(self.processed_file_path)) or overwrite: 
            print(f"Dataset will be processed and saved at {self.processed_file_path}")
            self.process()
        else : 
            self.load()

    def __len__(self):
        return len(self.input)
        
    def __getitem__(self, idx):
        input, target, length = self.input[idx], self.target[idx], self.length[idx]
        return input, target, length
        
    def process(self):
        if not os.path.exists(self.processed_dir_path):
            os.makedirs(self.processed_dir_path)

        start = time.time()
        self.data = read_all_shards(self.partition, self.raw_dir_path)

        # Filter to keep only the most frequent families
        if self.partition == 'train' and self.filter_fam :
            most_freq_fam = compute_most_freq_fam(self.data.family_accession.to_list(), self.n_fam)
            self.data = self.data[self.data['family_accession'].isin(most_freq_fam)]
            
        # Filter on sequence lengths
        if self.seq_lengths_bounds != (None, None): 
            self.data = self.data[self.data['seq_len'].between(*self.seq_lengths_bounds)]

        # Create mapping between families and their associated integer
        if self.partition == 'train':
            self.families_dict = generate_family_int_mapping(self.data['family_accession'].unique().tolist())
        else: 
            # keep only the samples that are present in the training set 
            self.data = self.data[self.data['family_accession'].isin(self.families_dict.keys())]

        #self.data = self.data.head(16) # Toy data

        # Encoding of sequences and family accession
        start = time.time()
        self.input = self.data['sequence'].apply(seq_to_onehot).to_list()
        self.target = self.data['family_accession'].apply(family_to_onehot, args=[self.families_dict]).to_list()
        self.length = [len(seq) for seq in self.input] # length of sequence may have changed after encoding due to removal of uncommon amino acid
        print(f"Elapsed time to process {self.partition} set: {time.time()-start}. It contains {len(self.data)} samples.")

        # Save processed data to avoid having to process it if already done 
        torch.save((self.input, self.target, self.length, self.families_dict), self.processed_file_path)

    
    def load(self):
        print(f"No processing required. {self.partition} set will be loaded from {self.processed_file_path}.")
        self.input, self.target, self.length, self.families_dict  = torch.load(self.processed_file_path)
        print(f"Loading done. It contains {len(self.input)} samples")



def custom_collate_fn(batch):
    # Custom function to collate the batch 
    inputs = [item[0] for item in batch]
    inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    targets = torch.stack([item[1] for item in batch]).float()
    lengths = default_collate([item[2] for item in batch])
    masks = (torch.arange(lengths.max())[None, :] < lengths[:, None]).unsqueeze(-1)

    # Turn [B, L, C] tensors into [B, C, L] - B=batch, L=length, C=channels
    inputs = inputs.permute((0,2,1)).float()
    masks = masks.permute((0,2,1))

    return inputs, targets, masks, lengths


# Custom BucketSampler to have batch of similar lengths in order to limit the difference in added padding for each sequence in the bucket
# taken from https://github.com/pytorch/pytorch/issues/46176
class BucketSampler(torch.utils.data.Sampler):

    def __init__(self, lengths, buckets=(50,500,50), shuffle=True, batch_size=32, drop_last=False):
        """ 
        BucketSampler. 
        Batches will be sampled within pools of samples that belong to the same length's bucket given the buckets parameters (len_min, len_max, len_step). 
        """
        
        super().__init__(lengths)
        
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        assert isinstance(buckets, tuple)
        bmin, bmax, bstep = buckets
        assert (bmax - bmin) % bstep == 0
        
        buckets = defaultdict(list)
        for i, length in enumerate(lengths):
            if length > bmin:
                bucket_size = min((length // bstep) * bstep, bmax)
                buckets[bucket_size].append(i)
                
        self.buckets = dict()
        for bucket_size, bucket in buckets.items():
            if len(bucket) > 0:
                self.buckets[bucket_size] = torch.tensor(bucket, dtype=torch.int, device='cpu')
        
        # call __iter__() to store self.length
        self.__iter__()
            
    def __iter__(self):
        
        if self.shuffle == True:
            for bucket_size in self.buckets.keys():
                self.buckets[bucket_size] = self.buckets[bucket_size][torch.randperm(self.buckets[bucket_size].nelement())]
                
        batches = []
        for bucket in self.buckets.values():
            curr_bucket = torch.split(bucket, self.batch_size)
            if len(curr_bucket) > 1 and self.drop_last == True:
                if len(curr_bucket[-1]) < len(curr_bucket[-2]):
                    curr_bucket = curr_bucket[:-1]
            batches += curr_bucket
            
        self.length = len(batches)
        
        if self.shuffle == True:
            random.shuffle(batches)
            
        return iter(batches)
    
    def __len__(self):
        return self.length


if __name__=='__main__':    
    dataset = PFAMDataset('train', None, overwrite=False)
    sampler = BucketSampler(dataset.length, buckets=(0, 2000, 400), shuffle=True, batch_size=32, drop_last=False)
    dl = torch.utils.data.DataLoader(dataset, batch_sampler=sampler, collate_fn=custom_collate_fn)
    for batch in dl : 
        inputs, targets, masks, lengths = batch 
        print(lengths)
        break 
    