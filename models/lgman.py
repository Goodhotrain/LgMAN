# lgman
import torch
import torch.nn as nn
from models.vit import Vit
from models.at2 import AudioTransformer
from models.CModalT import classif_head
from models.text_encoder import TextEncoder
from models.blip2qformer import MMCrossAttention
from transformers import BertTokenizer, BertConfig
import re
import random
from models.mbt_fusion import MBT
from models.Align import MultimodalAlignNet
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

class LgMAN(nn.Module):
    def __init__(self,
                 num_frames=8,
                 sample_size=224,
                 n_classes=8,
                 need_audio=False,
                 need_text=False,
                 audio_embed_size=256,
                 audio_n_segments=8,
                 text_embed_size=768,):
        super(LgMAN, self).__init__()

        self.need_audio = need_audio
        self.need_text = need_text
        self.audio_n_segments = audio_n_segments
        self.audio_embed_size = audio_embed_size
        self.num_frames = num_frames
        self.n_classes = n_classes
        # Vit
        self.tsformer = Vit(img_size=sample_size, num_classes=600, num_frames=num_frames, attention_type='divided_space_time')
        self.visual_embed_size = self.tsformer.model.embed_dim
        # Text
        if need_text:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.textencoder = TextEncoder()
        # Audio
        self.auformer = AudioTransformer(audio_n_segments, segment_len=audio_embed_size, num_classes=n_classes, embed_dim=768, depth=2)
        self.h = nn.Linear(self.visual_embed_size, self.visual_embed_size)
        self.model = MBT(2, 8, n_classes, 768)
        # Multi-modal Alignment Network
        self.align_net = MultimodalAlignNet(self.visual_embed_size, text_embed_size, self.visual_embed_size, aligned_dim=768)
        # self.align_net = ContrastiveAligner(self.visual_embed_size, text_embed_size, aligned_dim=768)
        self.cross = MMCrossAttention(layer_num=1)
        self.n=torch.nn.Sigmoid()
        self.head2 = classif_head(self.visual_embed_size, n_classes, drop=0.5)

    def forward(self, visual: torch.Tensor, audio: list, text: list):
        b = visual.shape[0]
        # Feature extraction
        # Visual Feature
        with torch.no_grad():
            x = rearrange(visual, 'b c t h w -> (b t) c h w',b=b,t=8).contiguous()
            visual = visual.contiguous()
            F_V, fv,ffv = self.tsformer(visual, x)
            # print(F_V.shape)
            V_tf = self.h(ffv)
            visual_embedded = rearrange(V_tf, '(b t) c -> b t c',b=b,t=8)
        # visual_embedded = rearrange(V_tf, '(b t) c -> b t c',b=b,t=8)
        if self.need_audio:
            # Audio Feature
            # [B x 8 x 256 x 32]
            a_f = []
            o = []
            t_f = []
            vat_f = []
            for num,( t,a_p) in enumerate(zip(text,audio)):
                with torch.no_grad():
                    _, f, _ = self.auformer(a_p)  # [B x 256]
                input_ids, attention_mask, masked_input_ids, masked_attention_mask, labels = self.text2tensor(t)
                F_T = self.textencoder(input_ids, attention_mask)
                # print(F_T.shape)
                loss_mlm = torch.tensor(0.0).cuda()
                # Align + Fusion
                loss_c =  self.align_net(F_V, text_cls, F_A)
                mlm_l, align_f = self.align_net.mlm(F_V[num].unsqueeze(0), F_A[num].unsqueeze(0) ,F_T.squeeze(0))
                o_i = self.cross(visual_embedded[num].unsqueeze(0), t_encoder_hidden_states = F_T, a_encoder_hidden_states=f.unsqueeze(0))
                o.append(o_i.squeeze(0)[0,:])
                loss_mlm += mlm_l
                vat_f.append(align_f)
                t_f.append(F_T.squeeze(0)[0,:])
            text_cls = torch.stack(t_f, dim=0)
            out_align = torch.stack(o, dim=0)
            F_A = torch.stack(a_f, dim=0)
            out_align = torch.stack(o, dim=0)
            # with torch.no_grad():
            output = self.model(fv, F_A.unsqueeze(1))      
            output = (0.2*output + 0.8*self.head2(out_align))
        return output, loss_c + loss_mlm

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

        # return input_ids, attention_mask, masked_input_ids.cuda(), masked_attention_mask.cuda(), labels.cuda()
        return input_ids, attention_mask