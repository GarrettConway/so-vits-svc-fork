import os
import random

import numpy as np
import torch
import torch.utils.data

from . import utils
from .modules.mel_processing import spectrogram_torch
from .utils import load_filepaths_and_text, load_wav_to_torch

import heapq


# import h5py


# Multi speaker version


class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
    1) loads audio, speaker_id, text pairs
    2) normalizes text and converts them to sequences of integers
    3) computes spectrograms from audio files.
    """

    def __init__(self, audiopaths, hparams):
        self.audiopaths = load_filepaths_and_text(audiopaths)
        self.max_wav_value = hparams.data.max_wav_value
        self.sampling_rate = hparams.data.sampling_rate
        self.filter_length = hparams.data.filter_length
        self.hop_length = hparams.data.hop_length
        self.win_length = hparams.data.win_length
        self.sampling_rate = hparams.data.sampling_rate
        self.use_sr = hparams.train.use_sr
        self.spec_len = hparams.train.max_speclen
        self.spk_map = hparams.spk

        random.seed(1234)
        random.shuffle(self.audiopaths)

    def get_audio(self, filename):
        filename = filename.replace("\\", "/")
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError(
                "{} SR doesn't match target {} SR".format(
                    sampling_rate, self.sampling_rate
                )
            )
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
        else:
            spec = spectrogram_torch(
                audio_norm,
                self.filter_length,
                self.sampling_rate,
                self.hop_length,
                self.win_length,
                center=False,
            )
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)

        spk = filename.split("/")[-2]
        spk = torch.LongTensor([self.spk_map[spk]])

        f0 = np.load(filename + ".f0.npy")
        f0, uv = utils.interpolate_f0(f0)
        f0 = torch.FloatTensor(f0)
        uv = torch.FloatTensor(uv)

        c = torch.load(filename + ".soft.pt")
        c = utils.repeat_expand_2d(c.squeeze(0), f0.shape[0])

        lmin = min(c.size(-1), spec.size(-1))
        assert abs(c.size(-1) - spec.size(-1)) < 3, (
            c.size(-1),
            spec.size(-1),
            f0.shape,
            filename,
        )
        assert abs(audio_norm.shape[1] - lmin * self.hop_length) < 3 * self.hop_length
        spec, c, f0, uv = spec[:, :lmin], c[:, :lmin], f0[:lmin], uv[:lmin]
        audio_norm = audio_norm[:, : lmin * self.hop_length]
        # if spec.shape[1] < 30:
        #     print("skip too short audio:", filename)
        #     return None
        if spec.shape[1] > 800:
            start = random.randint(0, spec.shape[1] - 800)
            end = start + 790
            spec, c, f0, uv = (
                spec[:, start:end],
                c[:, start:end],
                f0[start:end],
                uv[start:end],
            )
            audio_norm = audio_norm[:, start * self.hop_length: end * self.hop_length]

        return c, f0, spec, audio_norm, spk, uv

    def __getitem__(self, index):
        return self.get_audio(self.audiopaths[index][0])

    def __len__(self):
        return len(self.audiopaths)


class PreloadedTextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
    1) loads audio, speaker_id, text pairs
    2) normalizes text and converts them to sequences of integers
    3) computes spectrograms from audio files.
    """

    def __init__(self, audiopaths, hparams):
        self.audiopaths = load_filepaths_and_text(audiopaths)
        self.max_wav_value = hparams.data.max_wav_value
        self.sampling_rate = hparams.data.sampling_rate
        self.filter_length = hparams.data.filter_length
        self.hop_length = hparams.data.hop_length
        self.win_length = hparams.data.win_length
        self.sampling_rate = hparams.data.sampling_rate
        self.use_sr = hparams.train.use_sr
        self.spec_len = hparams.train.max_speclen
        self.spk_map = hparams.spk

        random.seed(1234)
        random.shuffle(self.audiopaths)

        # Hacky precompute code
        self.precompute = dict()

    def get_audio(self, filename):
        if filename not in self.precompute.keys():
            self.precompute[filename] = self._precompute(filename)
        c, f0, spec, audio_norm, spk, uv = self.precompute[filename]

        if spec.shape[1] > 800:
            start = random.randint(0, spec.shape[1] - 800)
            end = start + 790
            spec, c, f0, uv = (
                spec[:, start:end],
                c[:, start:end],
                f0[start:end],
                uv[start:end],
            )
            audio_norm = audio_norm[:, start * self.hop_length: end * self.hop_length]

        return c, f0, spec, audio_norm, spk, uv

    def _precompute(self, filename):
        filename = filename.replace("\\", "/")
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError(
                "{} SR doesn't match target {} SR".format(
                    sampling_rate, self.sampling_rate
                )
            )
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
        else:
            spec = spectrogram_torch(
                audio_norm,
                self.filter_length,
                self.sampling_rate,
                self.hop_length,
                self.win_length,
                center=False,
            )
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)

        spk = filename.split("/")[-2]
        spk = torch.LongTensor([self.spk_map[spk]])

        f0 = np.load(filename + ".f0.npy")
        f0, uv = utils.interpolate_f0(f0)
        f0 = torch.FloatTensor(f0)
        uv = torch.FloatTensor(uv)

        c = torch.load(filename + ".soft.pt")
        c = utils.repeat_expand_2d(c.squeeze(0), f0.shape[0])

        lmin = min(c.size(-1), spec.size(-1))
        assert abs(c.size(-1) - spec.size(-1)) < 3, (
            c.size(-1),
            spec.size(-1),
            f0.shape,
            filename,
        )
        assert abs(audio_norm.shape[1] - lmin * self.hop_length) < 3 * self.hop_length
        spec, c, f0, uv = spec[:, :lmin], c[:, :lmin], f0[:lmin], uv[:lmin]
        audio_norm = audio_norm[:, : lmin * self.hop_length]

        return c, f0, spec, audio_norm, spk, uv

    def __getitem__(self, index):
        # Wrap for replay
        index = index % len(self.audiopaths)
        return self.get_audio(self.audiopaths[index][0])

    def __len__(self):
        # return len(self.audiopaths)
        return len(self.audiopaths * 8840)


class WorstPerformingTextAudioSpeakerBatchLoader:
    """
    During training, the worst performing batches in each epoch get added here.
    """

    def __init__(self, num_saved=100, data_cache_file="toughData.test"):
        self.num_saved = num_saved
        self.data_cache_file = data_cache_file
        if os.path.exists(data_cache_file):
            print("Cache found for worst batches.")
            self.saved_data = torch.load(data_cache_file)
            # TODO add better handling for corrections in these cases
            assert type(self.saved_data) == list
            assert len(self.saved_data) == self.num_saved
            heapq.heapify(self.saved_data)
        else:
            self.saved_data = []

        self.counter = 0
        self.prepared_batches = None

    def add_batch(self, batch, loss_tensor):
        loss = loss_tensor.item()
        heap_item = (loss, batch)
        try:
            if len(self.saved_data) < self.num_saved:
                heapq.heappush(self.saved_data, heap_item)
            else:
                # Add the new batch and remove the lowest loss batch
                heapq.heappushpop(self.saved_data, heap_item)
        except:
            print("ERROR in updating heap for worst batches")

        self.counter += 1

    def prepare(self):
        self.prepared_batches = list(map(lambda x: x[1], self.saved_data))
        random.shuffle(self.prepared_batches)
        return self

    def clear(self):
        print("Clearing worst batches...")
        self.saved_data = []
        torch.save(self.saved_data, self.data_cache_file)

    def save(self):
        torch.save(self.saved_data, self.data_cache_file)
        print("Saved worst batches")

    def __getitem__(self, index):
        if self.prepared_batches is None:
            raise Exception("Not prepared for data loading. Call .prepare() first.")
        idx = index % len(self.prepared_batches)
        return self.prepared_batches[idx]

    def __len__(self):
        return len(self.prepared_batches)*10000


class TextAudioCollate:
    def __call__(self, batch):
        batch = [b for b in batch if b is not None]

        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[0].shape[1] for x in batch]), dim=0, descending=True
        )

        max_c_len = max([x[0].size(1) for x in batch])
        max_wav_len = max([x[3].size(1) for x in batch])

        lengths = torch.LongTensor(len(batch))

        c_padded = torch.FloatTensor(len(batch), batch[0][0].shape[0], max_c_len)
        f0_padded = torch.FloatTensor(len(batch), max_c_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][2].shape[0], max_c_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        spkids = torch.LongTensor(len(batch), 1)
        uv_padded = torch.FloatTensor(len(batch), max_c_len)

        c_padded.zero_()
        spec_padded.zero_()
        f0_padded.zero_()
        wav_padded.zero_()
        uv_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            c = row[0]
            c_padded[i, :, : c.size(1)] = c
            lengths[i] = c.size(1)

            f0 = row[1]
            f0_padded[i, : f0.size(0)] = f0

            spec = row[2]
            spec_padded[i, :, : spec.size(1)] = spec

            wav = row[3]
            wav_padded[i, :, : wav.size(1)] = wav

            spkids[i, 0] = row[4]

            uv = row[5]
            uv_padded[i, : uv.size(0)] = uv

        return c_padded, f0_padded, spec_padded, wav_padded, spkids, lengths, uv_padded
