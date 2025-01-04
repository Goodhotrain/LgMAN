from datasets.ekman6  import  ek6Dataset
from torch.utils.data import DataLoader


def get_ek6(opt, subset, transforms):
    spatial_transform, temporal_transform, target_transform, audio_transform = transforms
    return ek6Dataset(opt.video_path,
                      opt.audio_path,
                      opt.annotation_path,
                      opt.text_path,
                      subset,
                      opt.fps,
                      spatial_transform,
                      temporal_transform,
                      target_transform,
                      audio_transform,
                      need_audio= opt.need_audio)

def get_training_set(opt, spatial_transform, temporal_transform, target_transform, audio_transform):
    if opt.dataset == 'ek6':
        transforms = [spatial_transform, temporal_transform, target_transform, audio_transform]
        return get_ek6(opt, 'train', transforms)
    else:
        raise Exception

def get_validation_set(opt, spatial_transform, temporal_transform, target_transform, audio_transform):
    if opt.dataset == 'ek6':
        transforms = [spatial_transform, temporal_transform, target_transform, audio_transform]
        return get_ek6(opt, 'test', transforms)
    else:
        raise Exception


def get_test_set(opt, spatial_transform, temporal_transform, target_transform, audio_transform):
    if opt.dataset == 'ek6':
        transforms = [spatial_transform, temporal_transform, target_transform, audio_transform]
        return get_ek6(opt, 'test', transforms)
    else:
        raise Exception


def get_data_loader(opt, dataset, shuffle, batch_size=0):
    batch_size = opt.batch_size if batch_size == 0 else batch_size
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=opt.n_threads,
        pin_memory=True,
        drop_last=opt.dl,
    )