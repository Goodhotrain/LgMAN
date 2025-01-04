import argparse

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
                 default="/media/Harddisk/ghy/Mycode_e8",
                 type=str,
                 help='Global path of root directory'),
            dict(name="--video_path",
                 default="/media/Harddisk/Datasets/Micro_Video/MeiTu/video/",
                 type=str,
                 help='Global path of videos', ),
            dict(name="--audio_path",
                 default="/media/Harddisk/Datasets/Micro_Video/MeiTu/audio/",
                 type=str,
                 help='Global path of audios', ),
            dict(name="--text_path",
                 default='/media/Harddisk/ghy/Mycode_e8/preprocess/e8_title.json',                                                                
                 type=str,
                 help='Global path of title json file'),
            dict(name="--annotation_path",
                 default='/media/Harddisk/ghy/Mycode_e8/preprocess/em8.json',
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
                 default=8,
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
                 default=1e-1,
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
                default=100,
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
