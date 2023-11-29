from wordrain import generate_wordrain
from nltk.corpus import stopwords
import gensim

# Make manual collocations with troligt
def check_words(text):
    return_text = text.replace(" troligt", "_troligt")
    return return_text
    
# Example  for how to use generate_wordrain.generate_clouds
# It is expected to have subfolders in "swedish-climate-texts", which in turn contains txt files. It is also expected to be a stopword list called "swedish_stopwords_climate.txt".

WORD_SPACE_PATH = "../../../wordspaces/69/model.bin"
STOP_WORDS_LIST = "swedish_stopwords_climate.txt"
CORPUS_FOLDER = "swedish-climate-texts"
OUTPUT_FOLDER = "images_climate"

if __name__ == '__main__':
    from nltk.corpus import stopwords
    STOP_WORD_SET = set(stopwords.words('swedish'))

    with open(STOP_WORDS_LIST) as stopwords_file:
        stopwords = [el.strip() for el in stopwords_file.readlines()]
    all_stopwords = STOP_WORD_SET.union(stopwords)
    
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(WORD_SPACE_PATH, binary=True, unicode_errors='ignore')
        
    generate_wordrain.generate_clouds(CORPUS_FOLDER, word2vec_model, output_folder = OUTPUT_FOLDER, stopwords = all_stopwords, pre_process_method = check_words, fontsize=16)
