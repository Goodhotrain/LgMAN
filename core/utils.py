import os
import datetime
import shutil
import torch.nn as nn
from transforms.spatial import processing
from sklearn.metrics import f1_score

def local2global_path(opt):
    if opt.root_path != '':
        # if opt.debug:
        #     opt.result_path = "results/main"
        opt.result_path = os.path.join(opt.root_path, opt.result_paths)
        if opt.expr_name == '':
            now = datetime.datetime.now()
            now = now.strftime('result_%Y%m%d_%H%M%S')
            opt.result_path = os.path.join(opt.result_path, now)
        else:
            opt.result_path = os.path.join(opt.result_path, opt.expr_name)

            if os.path.exists(opt.result_path):
                shutil.rmtree(opt.result_path)
            os.mkdir(opt.result_path)

        opt.log_path = os.path.join(opt.result_path, "tensorboard")
        opt.ckpt_path = os.path.join(opt.result_path, "checkpoints")
        if not os.path.exists(opt.log_path):
            os.makedirs(opt.log_path)
        if not os.path.exists(opt.ckpt_path):
            os.mkdir(opt.ckpt_path)
    else:
        raise Exception


def get_spatial_transform(opt, mode):
    if mode == "train":
        return processing(size=opt.sample_size, is_aug=True, center=False)
    elif mode == "val":
        return processing(size=opt.sample_size, is_aug=False, center=True)
    elif mode == "test":
        return processing(size=opt.sample_size, is_aug=False, center=False)
    else:
        raise Exception


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def process_data_item(opt, data_item):
    visual, target, audio, text= data_item
    target = target.cuda()
    visual = visual.cuda()
    batch = visual.size(0)
    return visual, target, audio, text , batch

def run_model(opt, inputs, model, criterion, i=0, print_attention=True, period=30, return_attention=False):
    visual, target, audio, text = inputs
    if opt.mode == 'pretrain':
        # y_pred, loss_a = model(visual, audio, text)
        # loss = criterion(y_pred, target) + loss_a
        loss_c,loss_m = model(visual, audio, text)
        return loss_c,loss_m
    elif opt.mode == 'main':
        y_pred = model(visual, audio, text)
        loss = criterion(y_pred, target)
    return y_pred, loss

def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)
    values, indices = outputs.topk(k=1, dim=1, largest=True)
    pred = indices
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elements = correct.float()
    n_correct_elements = n_correct_elements.sum()
    n_correct_elements = n_correct_elements.item()
    return n_correct_elements / batch_size , indices.view(-1)
def compute_wa_f1(y_true, y_pred):
    """
    计算 Weighted Average F1 (WA-F1) 分数
    :param y_true: 真实标签 (list or array)
    :param y_pred: 预测标签 (list or array)
    :return: WA-F1 分数
    """
    return f1_score(y_true, y_pred, average='weighted')
