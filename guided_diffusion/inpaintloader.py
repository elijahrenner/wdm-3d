import os
import re
import pandas as pd
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class InpaintVolumes(Dataset):
    """Dataset returning MRI volumes and inpainting masks.

    Parameters
    ----------
    root_dir : str
        Base directory of the dataset.
    subset : str, optional
        ``'train'`` or ``'val'`` subset.
    img_size : int, optional
        Final side length of the returned volumes. Loaded data is padded to the
        largest dimension and downsampled to ``img_size`` if necessary.
    modalities : tuple, optional
        MRI modalities to load.
    normalize : callable, optional
        Preprocessing function applied to the returned volume.
    cache : bool, optional
        If ``True`` all data is preloaded into memory.
    """

    def __init__(
        self,
        root_dir: str,
        subset: str = "train",
        img_size: int = 256,
        modalities: tuple = ("T1w",),
        normalize=None,
        cache: bool = False,
    ):
        super().__init__()
        self.root_dir = os.path.expanduser(root_dir)
        self.subset = subset
        self.img_size = img_size
        self.modalities = modalities
        self.normalize = normalize or (lambda x: x)
        self.cases = self._index_cases()
        self.cache = None

        if cache:
            self.cache = [self._load_item(idx) for idx in range(len(self.cases))]

    # ------------------------------------------------------------
    def _index_cases(self):
        """Collect file paths for all cases."""
        df = pd.read_csv(f"{self.root_dir}/participants.tsv", sep="\t")

        # assign train/val split for FCD subjects
        fcd_df = df[df["group"] == "fcd"].copy()
        fcd_df = fcd_df.sample(frac=1, random_state=42).reset_index(drop=True)
        n_train = int(len(fcd_df) * 0.9)
        fcd_df.loc[: n_train - 1, "split"] = "train"
        fcd_df.loc[n_train:, "split"] = "val"
        df.loc[fcd_df.index, "split"] = fcd_df["split"]

        cases = []
        for pid in df[(df["split"] == self.subset) & (df["group"] == "fcd")][
            "participant_id"
        ]:
            case_dir = os.path.join(self.root_dir, pid, "anat")
            files = os.listdir(case_dir)
            img_dict = {}
            for mod in self.modalities:
                pattern = re.compile(rf"^{re.escape(pid)}.*{re.escape(mod)}\.nii\.gz$")
                matches = [f for f in files if pattern.match(f)]
                if not matches:
                    raise FileNotFoundError(f"Missing {mod} for {pid} in {case_dir}")
                img_dict[mod] = os.path.join(case_dir, matches[0])

            mask_matches = [
                f for f in files if re.match(rf"^{re.escape(pid)}.*roi\.nii\.gz$", f)
            ]
            if not mask_matches:
                raise FileNotFoundError(f"Missing mask for {pid} in {case_dir}")
            mask_path = os.path.join(case_dir, mask_matches[0])
            cases.append({"img": img_dict, "mask": mask_path, "name": pid})
        return cases

    # ------------------------------------------------------------
    def _pad_to_cube(self, vol, size, fill=0.0):
        """Symmetric 3-D pad to ``size`` cubed."""
        D, H, W = vol.shape[-3:]
        pad_D, pad_H, pad_W = (
            size - D,
            size - H,
            size - W,
        )
        pad = (
            pad_W // 2,
            pad_W - pad_W // 2,
            pad_H // 2,
            pad_H - pad_H // 2,
            pad_D // 2,
            pad_D - pad_D // 2,
        )
        return nn.functional.pad(vol, pad, value=fill)

    # ------------------------------------------------------------
    def _load_item(self, idx):
        rec = self.cases[idx]
        name = rec["name"]

        vols = []
        for mod in self.modalities:
            arr = (
                nib.load(rec["img"][mod]).get_fdata().astype(np.float32)
            )
            lo, hi = np.quantile(arr, [0.001, 0.999])
            arr = np.clip(arr, lo, hi)
            arr = (arr - lo) / (hi - lo + 1e-6)
            vols.append(torch.tensor(arr, dtype=torch.float32))
        first_mod = self.modalities[0]
        affine = nib.load(rec["img"][first_mod]).affine
        Y = torch.stack(vols, dim=0)

        mask_arr = nib.load(rec["mask"]).get_fdata().astype(np.uint8)
        M = torch.tensor(mask_arr, dtype=torch.float32).unsqueeze(0)
        M = (M > 0).to(Y.dtype)

        target_size = max(max(Y.shape[-3:]), self.img_size)
        if target_size % self.img_size != 0:
            target_size = ((target_size + self.img_size - 1) // self.img_size) * self.img_size

        Y = self._pad_to_cube(Y, target_size, fill=0.0)
        M = self._pad_to_cube(M, target_size, fill=0.0)
        if target_size != self.img_size:
            factor = target_size // self.img_size
            pool = nn.AvgPool3d(factor, factor)
            Y = pool(Y)
            M = pool(M)

        Y_void = Y * (1 - M)
        Y = self.normalize(Y)
        return Y, M, Y_void, name, affine

    # ------------------------------------------------------------
    def __getitem__(self, idx):
        if self.cache is not None:
            return self.cache[idx]
        return self._load_item(idx)

    # ------------------------------------------------------------
    def __len__(self):
        return len(self.cases)
