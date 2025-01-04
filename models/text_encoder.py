from transformers import BertTokenizer, BertForMaskedLM, BertConfig, BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F

class TextEncoder(nn.Module):
    def __init__(self, num_layers=2):
        super(TextEncoder, self).__init__()
        config = BertConfig.from_pretrained('bert-base-uncased')
        # config.num_hidden_layers = num_layers  # set Transformer layers
        self.bert = BertForMaskedLM.from_pretrained('bert-base-uncased', config=config)

    @torch.no_grad()
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return outputs.last_hidden_state
