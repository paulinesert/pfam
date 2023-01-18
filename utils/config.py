
import yaml

from argparse import Namespace
from ast import literal_eval as make_tuple


def open_config(config_path: str)-> Namespace : 
    # Open YAML config file given by a path config_path
    with open(config_path, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            raise exc
    return config 

def config_to_nested_args(config: dict)-> Namespace: 
    # Process YAML config as nested args 
    config_as_args = Namespace()
    update_args(config_as_args, config)
    return config_as_args

def update_args(args: Namespace, d: dict)-> Namespace: 
    # Recursive function to turn a dict d into a nested Namespace by updating a given Namespace args
    for k, v in d.items():
        if not isinstance(v, dict):
            setattr(args, k, v)
        else:
            ns = Namespace()
            update_args(ns, v)
            setattr(args, k, ns)

def process_PFAM_config(config: Namespace):
    # Read tuples in the form of strings and turn them into actual tuples with appropriate type
    setattr(config.data, 'seq_lengths_bounds', make_tuple(config.data.seq_lengths_bounds))
    setattr(config.data, 'bucket_sampler_buckets', make_tuple(config.data.bucket_sampler_buckets))
    return config

def generate_config(config_file_path: str)-> Namespace : 
    # Takes the path to a YAML config file, read it and turn it into a nested Namespace (args)
    config_yaml = open_config(config_file_path)
    config = config_to_nested_args(config_yaml)
    config = process_PFAM_config(config)
    return config 