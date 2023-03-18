from __future__ import annotations

from logging import getLogger
from pathlib import Path
from random import shuffle
from typing import Iterable, Literal

import librosa
import numpy as np
import torch
from joblib import Parallel, cpu_count, delayed
from tqdm import tqdm

from . import utils
from .utils import HUBERT_SAMPLING_RATE

LOG = getLogger(__name__)


def _process_one(filepath: Path, hubert_model, sampling_rate: int, hop_length: int,
                 device: Literal["cuda", "cpu"] = "cuda"):
    wav, sr = librosa.load(filepath, sr=sampling_rate)
    soft_path = filepath.parent / (filepath.name + ".soft.pt")
    if not soft_path.exists():
        wav16k = librosa.resample(
            wav, orig_sr=sampling_rate, target_sr=HUBERT_SAMPLING_RATE
        )
        wav16k = torch.from_numpy(wav16k).to(device)
        c = utils.get_hubert_content(hubert_model, wav_16k_tensor=wav16k)
        torch.save(c.cpu(), soft_path)
    f0_path = filepath.parent / (filepath.name + ".f0.npy")
    if not f0_path.exists():
        f0 = utils.compute_f0_dio(
            wav, sampling_rate=sampling_rate, hop_length=hop_length
        )
        np.save(f0_path, f0)


def _process_batch(filepaths: Iterable[Path], sampling_rate: int, hop_length: int, pos: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hubert_model = utils.get_hubert_model().to(device)

    for filepath in tqdm(filepaths, position=pos):
        _process_one(filepath, hubert_model, sampling_rate, hop_length, device)


def preprocess_hubert_f0(input_dir: Path | str, config_path: Path | str):
    input_dir = Path(input_dir)
    config_path = Path(config_path)
    hps = utils.get_hparams_from_file(config_path)
    sampling_rate = hps.data.sampling_rate
    hop_length = hps.data.hop_length

    filepaths = list(input_dir.glob("**/*.wav"))
    # Dual threading this until I can determine why this causes memory usage to explode and leak
    n_jobs = min(cpu_count(), len(filepaths) // 32 + 1, 2)
    shuffle(filepaths)
    filepath_chunks = np.array_split(filepaths, n_jobs)
    Parallel(n_jobs=n_jobs)(
        delayed(_process_batch)(chunk, sampling_rate, hop_length, pos) for (pos, chunk) in enumerate(filepath_chunks)
    )
