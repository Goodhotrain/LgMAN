import torch
import torch.utils.data as data
import os
import json
from torchvision.io import read_image, ImageReadMode
import torchaudio
from torchaudio import transforms
import torch
import random
from transformers import BertTokenizer
import re

from decord import VideoReader
import decord
import numpy as np
import random as rnd

decord.bridge.set_bridge("torch")

def load_video(video_path, n_frms=8, height=-1, width=-1, sampling="uniform", return_msg = False):
    decord.bridge.set_bridge("torch")
    vr = VideoReader(uri=video_path, height=height, width=width)

    vlen = len(vr)
    start, end = 0, vlen

    n_frms = min(n_frms, vlen)

    if sampling == "uniform":
        indices = np.arange(start, end, vlen / n_frms).astype(int).tolist()
    elif sampling == "headtail":
        indices_h = sorted(rnd.sample(range(vlen // 2), n_frms // 2))
        indices_t = sorted(rnd.sample(range(vlen // 2, vlen), n_frms // 2))
        indices = indices_h + indices_t
    else:
        raise NotImplementedError

    # get_batch -> T, H, W, C
    temp_frms = vr.get_batch(indices)
    # print(type(temp_frms))
    tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
    frms = tensor_frms.permute(0, 3, 1, 2).float()  # (T, C, H, W)

    if not return_msg:
        return frms

    fps = float(vr.get_avg_fps())
    sec = ", ".join([str(round(f / fps, 1)) for f in indices])
    # " " should be added in the start and end
    msg = f"The video contains {len(indices)} frames sampled at {sec} seconds. "
    return frms, msg

def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        return float(input_file.read().rstrip('\n\r'))

def preprocess_audio(audio_path='test.wav', transform = None):
    waveform, sample_rate = torchaudio.load(audio_path, normalize=True)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    waveform = waveform.squeeze(0)

    mfcc = transform(waveform)
    return mfcc.transpose(0, 1)

def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)

def get_video_names_and_annotations(video_path, audio_path, data, subset):
    '''
    :param data: json file
    :param subset: train, test, validation
    :return: video_names, annotations
    '''
    video_paths = []
    audio_paths = []
    annotations = []
    text = []
    a_path = '/media/Harddisk/Datasets/Micro_Video/MeiTu/audio/'
    id = []
    for key, value in data.items():
        # label = value['annotations']['label']
        # video_names.append('{}/{}'.format(label, key))
        if value['subset'] == subset:
            # v_p = os.path.join(video_path, value['id'])
            video_paths.append(value['video_path'])
            v_name = os.path.splitext(value['video_path'])[0].split('/')[-1]
            audio_paths.append(os.path.join(a_path, v_name+'.mp3'))
            text.append(value['title'])
            annotations.append(value['emotion'])
            id.append(key)
    return video_paths, audio_paths, text ,annotations, id


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map



def video_loader(video_dir_path, frame_indices):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, f'{i:06d}.jpg')
        assert os.path.exists(image_path), "image does not exists"
        video.append(read_image(image_path,mode =  ImageReadMode.RGB))
    return video


def count_frame(directory_path):
    entries = os.listdir(directory_path)
    jpg_files = [entry for entry in entries if entry.lower().endswith(".jpg")]
    jpg_file_count = len(jpg_files)
    return jpg_file_count

def get_default_video_loader():
    return video_loader

def pre_caption(caption,max_words):
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


class ek6Dataset(data.Dataset):
    def __init__(self,
                 video_path,
                 audio_path,
                 annotation_path,
                 text_path,
                 subset,
                 fps=30,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 audio_transform=None,
                 get_loader=get_default_video_loader,
                 need_audio=True):
        super(ek6Dataset, self).__init__()
        self.data= make_dataset(
            video_path,
            audio_path,
            text_path,
            annotation_path=annotation_path,
            subset=subset
        )

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.audio_transform = audio_transform
        self.loader = get_loader()
        self.fps = fps
        self.ORIGINAL_FPS = 30
        self.need_audio = need_audio
    #     self.transf = transforms.MFCC(
    #     sample_rate=44100,
    #     n_mfcc=20,
    #     melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 20, "center": False},
    # )

    def __getitem__(self, index):
        data_item = self.data[index]

        video_path = data_item['video']
        audio_path = data_item['audio']

        if self.need_audio:
            audios = audio_path
        else:
            audios = torch.zeros(1)

        ## Text Data
        text = data_item['text']

        # snippet = self.loader(video_path, snippets_frame_idx)
        snippet = load_video(video_path, n_frms=8, height=224, width=224, sampling="uniform")
        snippet = [self.spatial_transform(img) for img in snippet]
        random.seed()
        snippet = torch.stack(snippet, 0)
        snippet = snippet.permute(1, 0, 2, 3)

        target = self.target_transform(data_item)
        match int(target):
            case -5 | -4  :
                target = torch.tensor(+0)
            case -3 | -2:
                target = torch.tensor(+1)
            case -1 | 0 | 1:
                target = torch.tensor(+2)
            case 2 | 3 :
                target = torch.tensor(+3)
            case 4 | 5:
                target = torch.tensor(+4)
        # print(f'{target}')
        visualization_item = [data_item['video_id']]
        return snippet, target, audios, text

    def __len__(self):
        # return 10018
        return len(self.data)


def make_dataset(video_path, audio_path, text_path,annotation_path, subset):
    data = load_annotation_data(annotation_path)
    video_paths, audio_paths, texts, annotations, video_ids = get_video_names_and_annotations(video_path, audio_path, data, subset)

    dataset = []
    for i,(video_path, audio_path, text, annotation, video_id) in enumerate(zip(video_paths, audio_paths, texts, annotations, video_ids)):
        if i % 100 == 0:
            print("Dataset loading [{}/{}]".format(i, len(video_paths)))
        sample = {
            'video': video_path ,
            'audio': audio_path ,
            'video_id': video_id,
            'text': text,
            'label': annotation
        }
        dataset.append(sample)
    return dataset

