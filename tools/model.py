import torch
import os
from collections import OrderedDict

def freeze(*args):
    """
    Freeze the parameters of PyTorch models.

    Args:
        *args: Variable length argument list of PyTorch models.
    """
    for model in args:
        # Ensure that the argument is a PyTorch model
        if not isinstance(model, torch.nn.Module):
            raise TypeError("Argument must be a PyTorch model.")

        # Freeze the parameters of the model
        for param in model.parameters():
            param.requires_grad = False

def unfreeze(*args):
    """
    Freeze the parameters of PyTorch models.

    Args:
        *args: Variable length argument list of PyTorch models.
    """
    for model in args:
        # Ensure that the argument is a PyTorch model
        if not isinstance(model, torch.nn.Module):
            raise TypeError("Argument must be a PyTorch model.")

        # Freeze the parameters of the model
        for param in model.parameters():
            param.requires_grad = True

def print_network(net: torch.nn.Module):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    # print(net)
    print(f'Total number of learnable parameters: {num_params*4/(1024*1024):.6f} MB'  )


def load_state_dict(checkpoint_path, use_ema=False):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict_key = 'state_dict'
        if isinstance(checkpoint, dict):
            if use_ema and 'state_dict_ema' in checkpoint:
                state_dict_key = 'state_dict_ema'
        if state_dict_key and state_dict_key in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint[state_dict_key].items():
                # strip `module.` prefix
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v
            state_dict = new_state_dict
        elif 'model_state' in checkpoint:
            state_dict_key = 'model_state'
            new_state_dict = OrderedDict()
            for k, v in checkpoint[state_dict_key].items():
                # strip `model.` prefix
                name = k[6:] if k.startswith('model') else k
                new_state_dict[name] = v
            state_dict = new_state_dict
        else:
            state_dict = checkpoint
        print("Loaded {} from checkpoint '{}'".format(state_dict_key, checkpoint_path))
        return state_dict
    else:
        print("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()

def load_visual_pretrained(model:torch.nn.Module, model_file = ''):
    if model_file:
        d = load_state_dict(model_file)
        new_state_dict = OrderedDict()
        for k, v in d.items():
            if k.startswith('tsformer.'):
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)

def load_align_pretrained(model:torch.nn.Module, model_file = '/media/Harddisk/ghy/Mycode/results/debug2/result_20240318_154658/checkpoints'):
    # file_name = str(k_fold)+'_model_state.pth'
    # model_file = os.path.join(model_file, file_name)s
    d = load_state_dict(model_file)
    new_state_dict = OrderedDict()
    for k, v in d.items():
        if not k.startswith('head.'):
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=False)

def load_checkpoint(model:torch.nn.Module, checkpoint_path, use_ema=False, strict=True):
    state_dict = load_state_dict(checkpoint_path, use_ema)
    model.load_state_dict(state_dict, strict=strict)

def choose_save_checkpoint(model:torch.nn.Module):
    new_state_dict = OrderedDict()
    d= model.state_dict()
    for k,v in d.items():
        if k.startswith('head.') or k.startswith('cross.') or k.startswith('h.'):
            new_state_dict[k] = v
    return new_state_dict