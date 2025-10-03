import pandas as pd
from torch.utils.data import DataLoader
from dataloader import CSVDataloader

def test_loader():
    for chunk_id, chunk in enumerate(pd.read_csv('/mnt/self_discovery/isaacsaim_eskin_embedding/backbone_datasets/shuffle_datasets/training_datasets_DAI.csv', chunksize=65535)):
     chunk = pd.DataFrame(chunk.values, columns=chunk.columns)
     datasets = CSVDataloader(chunk)
     data_loader = DataLoader(datasets, batch_size=3, shuffle=True,
                              num_workers=4)
     for batch_idx, batch in enumerate(data_loader):
         # Data loading time

         # Move data to device
         X = batch['X']
         command = batch['command']
         target = batch['target']

if __name__ == "__main__":
    test_loader()