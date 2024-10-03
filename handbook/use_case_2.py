from wordrain import generate_wordrain
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
import sys
 
def get_stopwords():

    # Add numbers as stopwords
    stopword_list = [str(i) for i in range(0, 20000)]
    stopword_list += ["0" + i for i in stopword_list]
    
    # Swedish stop words included in NLTK
    stopword_list += stopwords.words('swedish')
    
    # Manually collected stopwords
    with open("handbook_sda_stopwords.txt") as stop_word_file:
        stopword_list +=[stopword.strip() for stopword in stop_word_file.readlines()]
    return stopword_list
    

def run_handbook_use_case_2(fontsize_falloff_rate, corpus_folder, word_to_vec_model_path):
    
    # Load the word2vec model
    word2vec_model =  KeyedVectors.load(word_to_vec_model_path)
        
    generate_wordrain.generate_clouds(corpus_folder= corpus_folder,
        word2vec_model = word2vec_model,
        output_folder = "use_case_2",
        mark_new_words=True,
        nr_of_words_to_show=600,
        show_number_of_occurrences=1000.0,
        min_tf_in_corpora=10,
        lowercase=False,
        nr_of_vertical_lines=40,
        fontsize_falloff_rate=fontsize_falloff_rate,
        stopwords=get_stopwords(),
        idf=False,
        scale_max_x_used=0.8,
        min_f_in_current_document=10)



# Start
########
try:
    corpus_folder = sys.argv[1]
    word_to_vec_model_path = sys.argv[2]
 
except IndexError:
    print("The script needs two arguments: The first is the path to the folder where the corpus is located. The second is the path to the word2vec model that is to be used.")
    exit()

# It is possible to do a fine-adjustment per text of how much the font size is to be decreased with decrease word prominence. To make the graphs look better, but not needed for functionality.
fontsize_falloff_rate = {generate_wordrain.DEFAULT_FONTSIZE_FALLOFF_RATE: 1.35, '1949': 1.05, '1952': 1.4,  '1954': 1.7, '1956': 1.2, '1957': 0.95, '1958': 1.4, '1959': 1.7, '1961': 1.4, '1964': 1.5, '1965': 1.31, '1968': 1.6, '1971': 1.7, '1972': 1.6, '1973': 1.55, '1974': 1.25, '1975': 1.2,  '1976': 1.35, '1977': 1.7, '1979': 1.55, '1980': 1.45, '1981': 1.2, '1982': 1.4, '1983': 1.7, '1984': 2.2, '1985': 1.3, '1986': 2.3, '1987': 1.7, '1988': 1.85, '1989': 1.75}

# Create one version with separate pdf files for each year
run_handbook_use_case_2(fontsize_falloff_rate = fontsize_falloff_rate, corpus_folder = corpus_folder, word_to_vec_model_path = word_to_vec_model_path)

# And another version where the graphs for all years are combined into one pdf file
#run_handbook_use_case_2(log_for_bar_height_dict = log_for_bar_height_dict, corpus_folder = corpus_folder, word_to_vec_model_path = word_to_vec_model_path, unified_graph=True)
