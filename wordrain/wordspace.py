import os
import json
import pickle

from numpy import dtype, frombuffer, float32

class MappedWord2Vec():
    def __init__(self, f, offsets, vector_size):
        self.f = f
        self.fileno = f.fileno()
        self.offsets = offsets
        self.vector_size = vector_size
    def __contains__(self, idx):
        return idx in self.offsets
    def __getitem__(self, idx):
        if idx not in self.offsets:
            return None
        size = self.vector_size * dtype(float32).itemsize
        offset = self.offsets[idx]
        b = os.pread(self.fileno, size, offset)
        if len(b) < size:
            return None
        vector = frombuffer(b, offset=0, count=self.vector_size, dtype=float32)

        return vector


def map_word2vec(filename):
    try:
        offsets = pickle.load(open(filename + ".pickleidx", "rb"))
    except FileNotFoundError:
        offsets = json.load(open(filename + ".jsonidx", "rt"))
    f = open(filename, "rb")
    header = f.readline().split()
    assert len(header) == 2
    vector_size = int(header[1])
    mapped = MappedWord2Vec(f, offsets, vector_size)
    return mapped
