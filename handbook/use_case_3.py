import wordrain
from wordrain import generate_wordrain
from wordrain import types
from wordrain import color

from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
import sys
 
# This is an example of what a color_map function you write yourself could look like.
# Here, we don't use it, but use a built-in function, call color.full_spectrum_map (see below)
"""
def color_map_function(cmap_value):
    cmap = matplotlib.colormaps['winter']
    color_base = cmap(cmap_value)
    midpoint = 0.5
    color_base_hsv = matplotlib.colors.rgb_to_hsv(color_base[0:3])
    color_base_hsv[1] = 0.6 #+ abs(midpoint-cmap_value)/3
    color_base_hsv[2] = 1.0 #- abs(midpoint-cmap_value)/5
    color_base = list(matplotlib.colors.hsv_to_rgb(color_base_hsv)) + [1.0]
    return color_base
"""

def transform_word_scores_to_diff(word_scores_per_graph_before_cutoff):
    new_word_scores_per_graph_before_cutoff = [] # To store the transformed scores to return
    scores_previous_graph = {} # To store the weights from the previous graph in the series
        
    # if the graphs are named with numers, 'word_scores_per_graph_before_cutoff' is sorted
    # according to these numbers. The last is the 'all' graph.
    for wordscores in word_scores_per_graph_before_cutoff[:-1]:
        
        # To store the word weights for the current graph in the series
        scores_current_graph = {}
            
        new_wordscores_for_graph = [] # The words with the new prominence values
        
        # ws is of type types.WordScore and looks, for instance, like this:
        ## WordScore(score=1.0, word='adekvat', force_inclusion=False, freq=1.0, relfreq=1.7250599458331175e-05)
        # 'score' is the score that will be used for sorting the words(, as well as 'force_inclusion'
        # which, if True, will make a word be sorted before those that have 'force_inclusion' set to False.)
        #
        # We here create new scores, from the difference between a words relfreq and the relfreq of previous year
        
        for ws in wordscores:
            
            relfreq_weighted_factor = ws.relfreq
            scores_current_graph[ws.word] = relfreq_weighted_factor # To save for next round in series
            
            score_from_diff = relfreq_weighted_factor - scores_previous_graph.get(ws.word, 0)
            if score_from_diff < 0:
                score_from_diff = 0 # Score cannot be negative
            new_wordscores_for_graph.append(types.WordScore(score_from_diff, ws.word, ws.force_inclusion, ws.freq, ws.relfreq))
            
        # To use for the next graph in the series, to be able to compare weight in current graph to previous
        scores_previous_graph = scores_current_graph
    
        new_word_scores_per_graph_before_cutoff.append(new_wordscores_for_graph)
        
    # for the last 'all' graph, which shows a summary of the entire corpus, only use the raw weights
    new_word_scores_per_graph_before_cutoff.append(word_scores_per_graph_before_cutoff[-1])

    return new_word_scores_per_graph_before_cutoff
    

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
    
def create_word_rain(fontsize_falloff_rate, corpus_folder, word_to_vec_model_path, unified_graph=False, compact=False):
    
    # Load a word2vec model trained on the corpus
    word2vec_model =  KeyedVectors.load(word_to_vec_model_path)
        
    wordrain.generate(corpus_folder=corpus_folder,
        word2vec_model=word2vec_model,
        output_folder = "use_case_3",
        mark_new_words=True,
        nr_of_words_to_show=400,
        show_number_of_occurrences=20,
        min_tf_in_corpora=10,
        lowercase=False,
        fontsize_falloff_rate=fontsize_falloff_rate,
        stopwords=get_stopwords(),
        idf=False,
        transform_word_scores=transform_word_scores_to_diff,
        min_f_in_current_document=10,
        y_extra_space_factor=1.1,
        color_map = color.full_spectrum_map,
        unified_graph=unified_graph,
        compact=True,
        nr_of_clusters=50,
        background_box=True,
        fixed_linewidth_down=None,
        show_words_outside_ylim=True)


# Start

try:
    corpus_folder = sys.argv[1]
    word_to_vec_model_path = sys.argv[2]
except IndexError:
    print("The script needs two arguments: The first is the path to the folder where the corpus is located. The second is the path to the word2vec model that is to be used.")
    exit()

# Use another font than the default font
wordrain.choose_fonts({
    wordrain.FONT_NORMAL: 'DejaVuSans.ttf',
    wordrain.FONT_EMPHASIZED: 'DejaVuSans-Bold.ttf', # Not used in this graph, but registered anyway just to show how to do it
    wordrain.FONT_NEWWORD: 'DejaVuSans-Oblique.ttf',
})
    
# It is possible to do a fine-adjustment per text of how much the font size is to be decreased with decrease word prominence. To make the graphs look better, but not needed for functionality.
fontsize_falloff_rate = {wordrain.DEFAULT_FONTSIZE_FALLOFF_RATE: 1.1, '1949': 0.9, '1953': 1.3, '1959': 0.9, '1960': 0.9,  '1962': 1.5, '1965': 1.0, '1966': 1.0, '1967': 0.9, '1968': 1.2, '1974': 1.4, '1976': 0.95, '1978': 0.9, '1982': 0.9, '1983': 0.9, '1985': 0.9, '1986': 1.3, '1988': 1.3, '1990': 0.8}

# Make one graph with individual images
create_word_rain(fontsize_falloff_rate = fontsize_falloff_rate, corpus_folder = corpus_folder, word_to_vec_model_path = word_to_vec_model_path)

# Also a unified graph with all images
create_word_rain(fontsize_falloff_rate = fontsize_falloff_rate, corpus_folder = corpus_folder, word_to_vec_model_path = word_to_vec_model_path, unified_graph=True)
