import sys
import json
import pickle

from numpy import float32, dtype

CHUNK_SIZE = 1 * 1024

def word2vec_entries(f, vocabulary_length, vector_size):
    vector_size_bytes = vector_size * dtype(float32).itemsize

    chunk = bytes()
    word_count = 0
    offset = 0
    while word_count < vocabulary_length:
        space_index = chunk.find(b" ")

        entrylength = space_index + 1 + vector_size_bytes
        if space_index == -1 or len(chunk) < entrylength:
            chunk += f.read(CHUNK_SIZE)
            continue

        yield (chunk[:space_index], offset + space_index + 1)
        chunk = chunk[entrylength:]
        offset += entrylength
        word_count += 1

def process_word2vec_file(f, vocabulary_length, vector_size, startoffset):
    index = {}
    for word, offset in word2vec_entries(f, vocabulary_length, vector_size):
        word = word.decode("utf-8", errors="strict").lstrip("\n")
        index[word] = offset + startoffset
    return index

def read_header(f):
    raw_header = f.readline()
    header = raw_header.decode("utf-8").split()
    assert len(header) == 2
    vocabulary_length = int(header[0])
    vector_size = int(header[1])
    return vocabulary_length, vector_size, len(raw_header)

def _load_word2vec_format_offsets(fname):
    with open(fname, 'rb') as fin:
        vocabulary_length, vector_size, startoffset = read_header(fin)
        print(vocabulary_length, vector_size, startoffset)

        return process_word2vec_file(fin, vocabulary_length, vector_size, startoffset)

def main(filename):
    print("start reading")
    offsets = _load_word2vec_format_offsets(filename)
    json.dump(offsets, open(filename + ".jsonidx", "wt"))
    pickle.dump(offsets, open(filename + ".pickleidx", "wb"))
    print("finished load offsets")

main(sys.argv[1])
