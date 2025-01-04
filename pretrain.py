# DEQ-fusion
import torch
import torch.nn as nn
import torchvision
from models.vit import TimeSformer
from models.at2 import AudioTransformer
from models.CModalT import CVAFM, classif_head
from models.text_encoder import TextEncoder
from models.blip2qformer import BertAttention, BertLayer, MMCrossAttention
from transformers import BertTokenizer, BertConfig
import re
import random
from models.mbt_fusion import MBT
from models.Align import MultimodalAlignNet, ContrastiveAligner
from ghy.model import freeze
from einops import rearrange


def pre_caption(caption, max_words):
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
            
    return caption

class Cformer(nn.Module):
    def __init__(self,
                 num_frames=8,
                 sample_size=224,
                 n_classes=8,
                 need_audio=False,
                 need_text=False,
                 audio_embed_size=256,
                 audio_n_segments=8,
                 text_embed_size=768,):
        super(Cformer, self).__init__()

        self.need_audio = need_audio
        self.need_text = need_text
        self.audio_n_segments = audio_n_segments
        self.audio_embed_size = audio_embed_size
        self.num_frames = num_frames
        self.n_classes = n_classes
        # self.model = DEQFusion(self.visual_embed_size, 2)
        # Vision TimeSformer
        model_file = '/media/Harddisk/ghy/models/TimeSformer_divST_8x32_224_K600.pyth'
        self.tsformer = TimeSformer(img_size=sample_size, num_classes=600, num_frames=num_frames, attention_type='divided_space_time', pretrained_model = model_file)
        self.visual_embed_size = self.tsformer.model.embed_dim
        # Text
        if need_text:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.textencoder = TextEncoder()
        # Audio
        self.auformer = AudioTransformer(audio_n_segments, segment_len=audio_embed_size, num_classes=n_classes, embed_dim=768, depth=2)
        self.h = nn.Linear(self.visual_embed_size, self.visual_embed_size)
        # self.model = MBT(2, 8, 8, 768)
        # Multi-modal Alignment Network
        self.align_net = MultimodalAlignNet(self.visual_embed_size, text_embed_size, self.visual_embed_size, aligned_dim=768)
        # self.align_net = ContrastiveAligner(self.visual_embed_size, text_embed_size, aligned_dim=768)
        self.cross = MMCrossAttention(layer_num=2)
        # self.n=torch.nn.Sigmoid()
        # self.head = classif_head(self.visual_embed_size, n_classes, drop=0.5)
        # self.head2 = nn.Linear(self.visual_embed_size, n_classes)
        # self.head = nn.Linear(self.visual_embed_size, n_classes)

    def forward(self, visual: torch.Tensor, audio: list, text: list):
        b = visual.shape[0]
        # Feature extraction
        # Visual Feature
        with torch.no_grad():
            x = rearrange(visual, 'b c t h w -> (b t) c h w',b=b,t=8).contiguous()
            visual = visual.contiguous()
            F_V, fv,ffv = self.tsformer(visual, x)
            # print(F_V.shape)
            # [B x 8 x 256 x 32]
            a_f = []
            for a_p in audio:
                a_F = self.auformer(a_p, MFCC = False)  # [B x 256]
                # torch.cuda.empty_cache()
                a_f.append(a_F)
            F_A = torch.stack(a_f, dim=0)
            # Text Feature
        if self.need_text:
            t_f = []
            vat_f = []
            loss_mlm = torch.tensor(0.0).cuda()
            for num, t in enumerate(text):
                input_ids, attention_mask, masked_input_ids, masked_attention_mask, labels = self.text2tensor(t)
                F_T = self.textencoder(input_ids, attention_mask)
                # mlm
                F_T_mask = self.textencoder(masked_input_ids, masked_attention_mask) # 1, len , 768
                mlm_l, align_f = self.align_net.mlm(F_V[num].unsqueeze(0), F_A[num].unsqueeze(0) ,F_T_mask.squeeze(0), labels)
                loss_mlm += mlm_l
                vat_f.append(align_f)
                t_f.append(F_T[:,0].squeeze(0))
            text_cls = torch.stack(t_f, dim=0)
            vat_f = torch.stack(vat_f, dim=0)
        # Alignment
        loss_c =  self.align_net(F_V, text_cls, F_A)
            # output = torch.stack(output, dim=0)
            # Text Feature
            # with torch.no_grad():
            #     output = self.model(fv, F_A.unsqueeze(1))
            #     output = torch.cat([F_V, output], dim=1)
                # output = self.head2(out_align) + 0.8 * self.head(output)
            # output = self.head2(output)
            # output = torch.max(self.n(output), self.n(self.head(out_align)))
            # output = (0.8*self.n(output) + 0.2*self.n(self.head(out_align)))
            # output = self.head(out_align)
            # output = self.head2(out_align)
        return loss_c, loss_mlm
    

    def text2tensor(self, text:str):
        text = pre_caption(text, 20)
        # Encode the text
        encoded_input = self.tokenizer(text, return_tensors="pt")
        input_ids = encoded_input["input_ids"].cuda()
        attention_mask = encoded_input["attention_mask"].cuda()

        tokenized_text = self.tokenizer.tokenize(text)
        masked_index = random.randint(0, len(tokenized_text) - 1)
        true_label = tokenized_text[masked_index]
        tokenized_text[masked_index] = 'MASK'
        masked_text = ' '.join(tokenized_text)

        # Encode the masked text
        encoded_masked_input = self.tokenizer(masked_text, return_tensors="pt")
        masked_input_ids = encoded_masked_input["input_ids"]
        masked_attention_mask = encoded_masked_input["attention_mask"]
        # Create labels tensor
        true_label_id = self.tokenizer.convert_tokens_to_ids([true_label])[0]
        m,n = masked_input_ids.shape
        labels = torch.full((m,n+2), -100)  # Initialize with -100 to ignore all tokens except masked
        labels[0, masked_index+2] = true_label_id # Set true label for the masked token

        return input_ids, attention_mask, masked_input_ids.cuda(), masked_attention_mask.cuda(), labels.cuda()
    
from core.loss import get_loss
from core.optimizer import get_optim
from core.utils import local2global_path, get_spatial_transform
from core.utils import AverageMeter
from datasets.dataset import get_training_set, get_validation_set, get_test_set, get_data_loader
from transforms.temporal import TSN
from transforms.audio import TSNAudio
from transforms.target import ClassLabel
import datetime
current_time = datetime.datetime.now()
print("Time:", current_time)
print("Notion: ")
from common.k_fold import read_csv

from ghy.model import load_visual_pretrained, load_align_pretrained
import torch
from tensorboardX import SummaryWriter
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import time
from core.utils import AverageMeter, process_data_item, run_model, calculate_accuracy

import os
import time
import torch
from core.utils import AverageMeter, process_data_item, run_model, calculate_accuracy, compute_wa_f1
from ghy.model import choose_save_checkpoint

def val_epoch(epoch, data_loader, model, criterion, opt, writer, optimizer):
    # k, epoch, valid_acc = e
    print("# ---------------------------------------------------------------------- #")
    print('Validation at epoch {}'.format(epoch))
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_c = AverageMeter()
    losses_m = AverageMeter()
    accuracies = AverageMeter()
    end_time = time.time()
    
    all_preds = []
    all_labels = []
    for i, data_item in enumerate(data_loader):
        visual, target, audio,text, batch_size = process_data_item(opt, data_item)
        data_time.update(time.time() - end_time)
        with torch.no_grad():
            loss_c, loss_m  = run_model(opt, [visual, target, audio, text], model, criterion, i)

        # acc, pre = calculate_accuracy(output, target)

        losses_c.update(loss_c.item(), batch_size)
        losses_m.update(loss_m.item(), batch_size)
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        # all_preds.extend(pre.cpu().numpy().tolist())
        # all_labels.extend(target.cpu().numpy().tolist())
    # wa_f1_score = compute_wa_f1(all_labels, all_preds)

    writer.add_scalar(f'val/loss_c', losses_c.avg, epoch)
    writer.add_scalar(f'val/loss_m', losses_m.avg, epoch)
    # print("Val loss: {:.4f}".format(losses.avg))
    print("Val acc: {:.4f}".format(accuracies.avg))
    print("Train loss: {:.4f}".format(losses_c.avg))
    print("Train loss: {:.4f}".format(losses_m.avg))
    # return max(accuracies.avg, valid_acc)



def train_epoch(e, data_loader, model, criterion, optimizer, opt, class_names, writer):
    k, epoch = e
    print("# ---------------------------------------------------------------------- #")
    print('Training at epoch {}'.format(epoch))
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_c = AverageMeter()
    losses_m = AverageMeter()
    accuracies = AverageMeter()
    end_time = time.time()

    for i, data_item in enumerate(data_loader):
        visual, target, audio, text, batch_size = process_data_item(opt, data_item)
        data_time.update(time.time() - end_time)
        loss_c, loss_m = run_model(opt, [visual, target, audio, text], model, criterion, i, print_attention=False)
        # acc,_ = calculate_accuracy(output, target)
        losses_c.update(loss_c.item(), batch_size)
        losses_m.update(loss_m.item(), batch_size)
        # accuracies.update(acc, batch_size)

        # Backward and optimize0
        optimizer.zero_grad()
        (0.2*loss_m + 0.8*loss_c).backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

    # ---------------------------------------------------------------------- #
    print("Epoch Time: {:.2f} min".format(batch_time.sum/ 60))
    print("Total Time: {:.2f} h".format(batch_time.sum * opt.n_epochs*5/ (3600*4)))
    print("Train loss: {:.4f}".format(losses_c.avg))
    print("Train loss: {:.4f}".format(losses_m.avg))


    writer.add_scalar(f'train/loss_c', losses_c.avg, epoch)
    writer.add_scalar(f'train/loss_m', losses_m.avg, epoch)


def generate_model(opt, k_fold:int):
    model = Cformer(
        num_frames=opt.n_frames,
        sample_size=opt.sample_size,
        n_classes=opt.n_classes,
        audio_embed_size=opt.audio_embed_size,
        audio_n_segments=opt.audio_n_segments,
        need_audio=opt.need_audio,
        need_text=opt.need_text,
    )
    assert opt.mode in ['pretrain', 'main'], 'mode should be pretrain or main'
    # if opt.mode == 'pretrain' and opt.visual_pretrained:
    #    load_visual_pretrained(model, opt.visual_pretrained)
    # # load_visual_pretrained(model, opt.visual_pretrained)
    
    # load_align_pretrained(model, k_fold, model_file='/media/Harddisk/ghy/Mycode_e8/results/debug2/result_20240428_210721/checkpoints/1_40model_state.pth')
    # load_align_pretrained(model, model_file='/media/HardDisk/ghy/Mycode_e8/results/debug/result_20240615_214538/checkpoints/1_30model_state.pth')
    # load_align_pretrained(model, model_file='/media/HardDisk/ghy/Mycode_e8/results/debug2/result_20240712_194651/checkpoints/1_20model_state.pth')
    # load_visual_pretrained(model, opt.visual_pretrained)
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
                 default="/media/Backup/ghy/MTSVRC",
                 type=str,
                 help='Global path of root directory'),
            dict(name='--result_paths',
                 default="results/pretrain",
                 type=str,
                 help='local path of results directory'),   
            dict(name="--video_path",
                 default="/media/Harddisk/Datasets/Micro_Video/MeiTu/video/",
                 type=str,
                 help='Global path of videos', ),
            dict(name="--audio_path",
                 default="/media/Harddisk/Datasets/Micro_Video/MeiTu/audio/",
                 type=str,
                 help='Global path of audios', ),
            dict(name="--text_path",
                 default='/media/Harddisk/ghy/C/MTSVRC/preprocess/mtsvrc_title.json',                                                                
                 type=str,
                 help='Global path of title json file'),
            dict(name="--annotation_path",
                 default='/media/Harddisk/ghy/C/MTSVRC/preprocess/mtsvrc_title.json',
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
                 default=3,
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
                 default='pretrain',
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
        # s_acc = 0.
        for i in range(1, opt.n_epochs + 1) :
            train_epoch((k_fold, i), train_loader, model, criterion, optimizer, opt, None, writer)
            val_epoch( i , val_loader, model, criterion, opt, writer, optimizer)
        # total_acc.update(s_acc)
    # print(f"Total Acc: {total_acc.avg:.4f}")
    writer.close()