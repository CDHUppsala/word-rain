import wordrain
import nltk.corpus
import gensim

# Example showing how to use wordrain.generate.
# This code expects text collections as subfolders in "swedish-climate-texts" with each subfolder containing txt files.

WORD_SPACE_PATH = "./model-sv.bin"
CORPUS_FOLDER = "swedish-climate-texts"
OUTPUT_FOLDER = "images_climate"

stopwords = nltk.corpus.stopwords.words('swedish')

word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(WORD_SPACE_PATH, binary=True, unicode_errors='ignore')
            
wordrain.generate(CORPUS_FOLDER, word2vec_model, output_folder=OUTPUT_FOLDER, stopwords=stopwords, unified_graph=True, compact=True)
