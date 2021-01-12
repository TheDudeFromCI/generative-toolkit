from math import floor
from itertools import repeat

from torch.cuda import FloatTensor
from torch.utils.data import DataLoader
from torch.autograd.variable import Variable

from gentool.database.image_dataset import ImageDataset


def image_dataloader(folders, batch_size, transform, format='RGB'):
    dataloader = DataLoader(ImageDataset(folders, transform, format), batch_size=batch_size,
                            shuffle=True, num_workers=8, drop_last=True, persistent_workers=True)

    for loader in repeat(dataloader):
        for _, batch in enumerate(loader):
            batch = Variable(batch.type(FloatTensor))
            yield batch
