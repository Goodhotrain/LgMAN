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
from models.Align import MultimodalAlignNet, ContrastiveAligner


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
        
        # Multi-modal Alignment Network
        # self.align_net = MultimodalAlignNet(self.visual_embed_size, text_embed_size, self.visual_embed_size, aligned_dim=768)
        # self.align_net = ContrastiveAligner(self.visual_embed_size, text_embed_size, aligned_dim=768)
        self.cross = MMCrossAttention(layer_num=6)
        # Fusion
        # self.av_fc = nn.Linear(self.visual_embed_size+ text_embed_size,self.n_classes)
        # self.av_fc = classif_head(self.visual_embed_size + self.visual_embed_size + text_embed_size, n_classes)
        
        # Cross-temporal Vision-Audio Fusion Mudule
        # self.cav_fm = CVAFM(self.visual_embed_size, num_heads=8)
        
        # classif_head
        # self.head_v = nn.Linear(self.visual_embed_size, n_classes)
        self.attn1 = nn.Parameter(torch.tensor(0.5))
        self.attn2 = nn.Parameter(torch.tensor(0.5))
        self.head = classif_head(self.visual_embed_size, n_classes, drop=0.5)
        self.head2 = classif_head(self.visual_embed_size, n_classes, drop=0.5)

    def forward(self, visual: torch.Tensor, audio: list, text: list):
        visual = visual.contiguous()
        # Feature extraction
        # Visual Feature
        with torch.no_grad():
            F_V, V_tf= self.tsformer(visual)
            # print(V_tf.shape)
        if self.need_audio:
            # Audio Feature
            # [B x 8 x 256 x 32]
            a_f = []
            a_ft = []
            for a_p in audio:
                with torch.no_grad():
                    a_F, f, _ = self.auformer(a_p)  # [B x 256]
                    # a_F = self.auformepr(a_, MFCC = False)  # [B x 256]
                # torch.cuda.empty_cache()
                a_f.append(a_F)
                a_ft.append(f)
            F_A = torch.stack(a_f, dim=0)
            audio_temporal = torch.stack(a_ft, dim=0)
            
            # Text Feature
            if self.need_text:
                t_f = []
                vat_f = []
                o = []
                for num, t in enumerate(text):
                    input_ids, attention_mask, masked_input_ids, masked_attention_mask, labels = self.text2tensor(t)
                    F_T = self.textencoder(input_ids, attention_mask)
                    # print(F_T.shape)
                    # cross attention
                    o_i = self.cross(F_T, t_encoder_hidden_states =V_tf[num].unsqueeze(0), a_encoder_hidden_states=audio_temporal[num].unsqueeze(0))
                    o.append(o_i.squeeze(0)[0,:])
                    # align + fusion
                    # print(F_T.shape)
                    # with torch.no_grad():
                    #     mlm_l, align_f = self.align_net.mlm(F_V[num].unsqueeze(0), F_A[num].unsqueeze(0) ,F_T.squeeze(0))
                    # vat_f.append(align_f)
                    t_f.append(F_T.squeeze(0)[0,:])
                text_cls = torch.stack(t_f, dim=0)
                # vat_f = torch.stack(vat_f, dim=0)
                out_align = torch.stack(o, dim=0)
            # Cat Fusion
            # fSCTA = torch.cat([F_V, F_A], dim=1)
            # output = self.head(fSCTA)
            # CVAFM Fusion
            # output = self.cav_fm(V_tf, audio_temporal)
            # output = torch.cat([F_V, vat_f, output], dim=1)
            output = self.head2(out_align) + self.head(F_V)
        else:
            # t_f = []
            # vat_f = []
            # loss_mlm = torch.tensor(0.0).cuda()
            # for num, t in enumerate(text):
            #         input_ids, attention_mask, masked_input_ids, masked_attention_mask, labels = self.text2tensor(t)
            #         F_T = self.textencoder(input_ids, attention_mask)
            #         # mlm
            #         F_T_mask = self.textencoder(masked_input_ids, masked_attention_mask) # 1, len , 768
            #         mlm_l, align_f = self.align_net.mlm_vt(F_V[num].unsqueeze(0), F_T_mask.squeeze(0), labels)
            #         loss_mlm += mlm_l
            #         vat_f.append(align_f)
            #         t_f.append(F_T[:,0].squeeze(0))
            # text_cls = torch.stack(t_f, dim=0)
            # vat_f = torch.stack(vat_f, dim=0)
            # # Text + Vision
            # output = torch.cat([F_V, vat_f], dim=1)
            output = self.head(F_V)
        return output

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

        # Encode the masked text]
        encoded_masked_input = self.tokenizer(masked_text, return_tensors="pt")
        masked_input_ids = encoded_masked_input["input_ids"]
        masked_attention_mask = encoded_masked_input["attention_mask"]
        # Create labels tensor
        true_label_id = self.tokenizer.convert_tokens_to_ids([true_label])[0]
        m,n = masked_input_ids.shape
        labels = torch.full((m,n+2), -100)  # Initialize with -100 to ignore all tokens except masked
        labels[0, masked_index+2] = true_label_id # Set true label for the masked token

        return input_ids, attention_mask, masked_input_ids.cuda(), masked_attention_mask.cuda(), labels.cuda()
