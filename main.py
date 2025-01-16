from core.loss import get_loss
from core.optimizer import get_optim
from core.utils import local2global_path, get_spatial_transform
from core.utils import AverageMeter
from datasets.dataset import get_training_set, get_validation_set, get_test_set, get_data_loader
from transforms.temporal import TSN
from transforms.audio import TSNAudio
from transforms.target import ClassLabel
from models.lgman import LgMAN
import datetime
current_time = datetime.datetime.now()
print("Time:", current_time)
print("Notion: ")
from common.k_fold import read_csv
from train import train_epoch
from validation import val_epoch
from tools.model import load_visual_pretrained, load_align_pretrained
import torch
from tensorboardX import SummaryWriter
import argparse
import os

def generate_model(opt, k_fold:int):
    model = LgMAN(
        num_frames=opt.n_frames,
        sample_size=opt.sample_size,
        n_classes=opt.n_classes,
        audio_embed_size=opt.audio_embed_size,
        audio_n_segments=opt.audio_n_segments,
        need_audio=opt.need_audio,
        need_text=opt.need_text,
    )
    assert opt.mode in ['pretrain', 'main'], 'mode should be pretrain or main'
    model = model.cuda()
    return model, model.parameters()

torch.random.manual_seed(99)

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    # print(net)
    print(f'Total number of learnable parameters: {num_params*4/(1024*1024):.6f} MB')

def load_pretrained(model, optimizer, args):
    print("===> Setting Pretrained Checkpoint")
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("===> loading models '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print("===> no models found at '{}'".format(args.pretrained))
        return checkpoint['epoch']
    else:
        return 1

def parse_opts():
    parser = argparse.ArgumentParser()
    arguments = {
        'coefficients': [
            dict(name='--lambda_0',
                 default='0.5',
                 type=float,
                 help='Penalty Coefficient that Controls the Penalty Extent in PCCE'),
        ],
        'paths': [
            dict(name='--root_path',
                 default="",
                 type=str,
                 help='Global path of root directory'),
            dict(name='--result_paths',
                 default="results/main",
                 type=str,
                 help='local path of results directory'),                 
            dict(name="--video_path",
                 default="MeiTu/video/",
                 type=str,
                 help='Global path of videos', ),
            dict(name="--audio_path",
                 default="MeiTu/audio/",
                 type=str,
                 help='Global path of audios', ),
            dict(name="--text_path",
                 default='mtsvrc_title.json',                                                                
                 type=str,
                 help='Global path of title json file'),
            dict(name="--annotation_path",
                 default='mtsvrc_title.json',
                 type=str,
                 help='Global path of annotation file'),
            dict(name="--result_path",
                 default='results',
                 type=str,
                 help="Local path of result directory"),
            dict(name='--expr_name',
                 type=str,
                 default=''),
        ],
        'core': [
            dict(name='--batch_size',
                 default=8,
                 type=int,
                 help='Batch Size'),
            dict(name='--sample_size',
                 default=224,
                 type=int,
                 help='Heights and width of inputs'),
            dict(name='--n_classes',
                 default=5,
                 type=int,
                 help='Number of classes'),
            dict(name='--n_frames',
                 default=8,
                 type=int),
            dict(name='--loss_func',
                 default='ce',
                 type=str,
                 help='ce | pcce_ve8'),
            dict(name='--learning_rate',
                 default=1e-4,
                 type=float,
                 help='Initial learning rate',),
            dict(name='--weight_decay',
                 default=0.0001,
                 type=float,
                 help='Weight Decay'),
            dict(name='--fps',
                 default=30,
                 type=int,
                 help='fps'),
            dict(name='--mode',
                 default='main',
                 type=str,
                 help='choose pretrain or main or visual pretrain'),
        ],
        'network': [
            {
                'name': '--audio_embed_size',
                'default': 256,
                'type': int,
            },
            {
                'name': '--audio_n_segments',
                'default': 8,
                'type': int,
            }
        ],

        'common': [
            dict(name='--need_audio',
                 type=bool,
                 default=True,
                 ),
            dict(name='--need_text',
                 type=bool,
                 default=True,
                 ),
            dict(name='--dataset',
                 type=str,
                 default='ek6',
                 ),
            dict(name='--debug',
                 default=True,
                 action='store_true'),
            dict(name='--dl',
                 action='store_true',
                 default=False,
                 help='drop last'),
            dict(
                name='--n_threads',
                default = 8,
                type=int,
                help='Number of threads for multi-thread loading',
            ),
            dict(
                name='--n_epochs',
                default=200,
                type=int,
                help='Number of total epochs to run',
            ),
            dict(
                name='--pretrained',
                default='',
                type=str,
                help='directory of pretrained model',
            ),
            dict(
                name='--visual_pretrained',
                default='',
                type=str,
                help='directory of pretrained TimeSformer model',
            ),
        ]
    }
    for group in arguments.values():
        for argument in group:
            name = argument['name']
            del argument['name']
            parser.add_argument(name, **argument)

    args = parser.parse_args([])
    return args



if __name__ == "__main__":
    opt = parse_opts()
    local2global_path(opt)
    print('learnable word embedding')
    total_acc = AverageMeter()
    for k_fold, _ in enumerate(read_csv(k=1), 1):
        print(f"# -----------------------------------<{k_fold}> fold----------------------------------- #")
        model, parameters = generate_model(opt, k_fold)
        print_network(model)

        criterion = get_loss(opt)
        criterion = criterion.cuda()

        optimizer = get_optim(opt, parameters, 'sgd')
        start_epoch = load_pretrained(model, optimizer, opt)
        writer = SummaryWriter(logdir = opt.log_path)

        # train
        spatial_transform = get_spatial_transform(opt, 'train')
        temporal_transform = TSN(n_frames=opt.n_frames, center=False)
        target_transform = ClassLabel()
        audio_transform = TSNAudio(n_frames=opt.n_frames, center=False)
        training_data = get_training_set(opt, spatial_transform, temporal_transform, target_transform, audio_transform)
        train_loader = get_data_loader(opt, training_data, shuffle=True)

        # validation
        spatial_transform = get_spatial_transform(opt, 'test')
        temporal_transform = TSN(n_frames=opt.n_frames, center=False)
        target_transform = ClassLabel()
        audio_transform = TSNAudio(n_frames=opt.n_frames, center=False)
        validation_data = get_validation_set(opt, spatial_transform, temporal_transform, target_transform, audio_transform)
        val_loader = get_data_loader(opt, validation_data, shuffle=False)
        s_acc = 0.
        for i in range(1, opt.n_epochs + 1) :
            train_epoch((k_fold, i), train_loader, model, criterion, optimizer, opt, None, writer)
            s_acc = val_epoch((k_fold, i, s_acc), val_loader, model, criterion, opt, writer, optimizer)
        total_acc.update(s_acc)
    print(f"Total Acc: {total_acc.avg:.4f}")
    writer.close()