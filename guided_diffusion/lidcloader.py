import torch
import torch.nn as nn
import torch.utils.data
import os
import os.path
import nibabel


class LIDCVolumes(torch.utils.data.Dataset):
    def __init__(self, directory, test_flag=False, normalize=None, mode='train', img_size=256, cache=False):
        """LIDC-IDRI dataset loader.

        Parameters
        ----------
        directory : str
            Path to the dataset root directory.
        test_flag : bool, optional
            If ``True`` only image volumes are returned.
        normalize : callable, optional
            Optional preprocessing function applied to the returned volume.
        mode : str, optional
            ``'train'`` or ``'fake'``.
        img_size : int, optional
            Side length of the returned cubic volume. Images are padded to a
            256³ cube and downsampled if ``img_size`` is smaller than 256.
        """
        super().__init__()
        self.mode = mode
        self.directory = os.path.expanduser(directory)
        self.normalize = normalize or (lambda x: x)
        self.test_flag = test_flag
        self.img_size = img_size
        self.database = []
        self.cache = None

        if not self.mode == 'fake':
            for root, dirs, files in os.walk(self.directory):
                # if there are no subdirs, we have a datadir
                if not dirs:
                    files.sort()
                    datapoint = dict()
                    # extract all files as channels
                    for f in files:
                        datapoint['image'] = os.path.join(root, f)
                    if len(datapoint) != 0:
                        self.database.append(datapoint)
        else:
            for root, dirs, files in os.walk(self.directory):
                for f in files:
                    datapoint = dict()
                    datapoint['image'] = os.path.join(root, f)
                    self.database.append(datapoint)

        if cache:
            self.cache = [self._load_item(i) for i in range(len(self.database))]

    def _load_item(self, x):
        filedict = self.database[x]
        name = filedict['image']
        nib_img = nibabel.load(name)
        out = nib_img.get_fdata().astype(np.float32)

        if not self.mode == 'fake':
            out = torch.tensor(out, dtype=torch.float32)

            image = torch.zeros(1, 256, 256, 256, dtype=torch.float32)
            image[:, :, :, :] = out

            if self.img_size != 256:
                assert 256 % self.img_size == 0, "img_size must divide 256"
                factor = 256 // self.img_size
                downsample = nn.AvgPool3d(kernel_size=factor, stride=factor)
                image = downsample(image)
        else:
            image = torch.tensor(out, dtype=torch.float32)
            image = image.unsqueeze(dim=0)

        # normalization
        image = self.normalize(image)

        if self.mode == 'fake':
            return image, name
        else:
            return image

    def __getitem__(self, x):
        if self.cache is not None:
            return self.cache[x]
        return self._load_item(x)

    def __len__(self):
        return len(self.database)
