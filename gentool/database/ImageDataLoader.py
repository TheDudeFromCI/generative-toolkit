from math import floor

from torch.cuda import FloatTensor
from torch.utils.data import DataLoader
from torch.autograd.variable import Variable

from gentool.database.image_dataset import ImageDataset


def image_dataloader(folders, batch_size, transform, format='RGB'):
    dataloader = DataLoader(ImageDataset(folders, transform, format, infinite=True), batch_size=batch_size,
                            shuffle=True, num_workers=8, prefetch_factor=floor(batch_size*2/8),
                            persistent_workers=True)

    for _, batch in enumerate(dataloader):
        batch = Variable(batch[0].type(FloatTensor))
        yield batch
