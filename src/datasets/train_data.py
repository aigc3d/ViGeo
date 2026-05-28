import torch
from mmengine import DATASETS

@DATASETS.register_module()
class train_data(torch.utils.data.Dataset):
    def __init__(
        self,
        datasets,
    ):
        super().__init__()
        self.datasets = [
            DATASETS.build(dataset) for dataset in datasets]
        
        self.data_length = sum(len(dataset) for dataset in self.datasets)
    
    def __len__(self):
        return self.data_length

    def __getitem__(self, index):
        data_idx, seq_idx, seq_len, seed, tgt_h, tgt_w = index 
        
        sample_dataset = self.datasets[data_idx]
        data_name = getattr(sample_dataset, 'data_name', f'Dataset_{data_idx}')
        
        sample = {
            "seed": seed,
            "tgt_height": tgt_h,
            "tgt_width": tgt_w,
            "data_name": data_name,
        }

        video_data = sample_dataset.sample_video(seq_idx, seq_len, seed)
        sample.update(video_data)
        sample = sample_dataset.pipeline(sample)
        return sample
