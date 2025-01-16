import torch.nn as nn
from models.ctmmfn2 import Cformer
from tools.model import load_visual_pretrained, load_align_pretrained

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
    load_align_pretrained(model, k_fold, model_file='/media/Harddisk/ghy/Mycode_e8/results/debug2/result_20240706_171409/checkpoints/1_10model_state.pth')
    # load_visual_pretrained(model, opt.visual_pretrained)
    model = model.cuda()
    return model, model.parameters()
