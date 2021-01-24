from math import floor
from itertools import repeat

from torch.cuda import FloatTensor
from torch.utils.data import DataLoader
from torch.autograd.variable import Variable

from gentool.database.image_dataset import ImageDataset


def image_dataloader(folders, batch_size, transform, format='RGB', workers=4):
    dataloader = DataLoader(ImageDataset(folders, transform, batch_size, format), batch_size=batch_size,
                            num_workers=workers, persistent_workers=True, prefetch_factor=8)

    for loader in repeat(dataloader):
        for _, batch in enumerate(loader):
            batch = Variable(batch.type(FloatTensor))
            yield batch
