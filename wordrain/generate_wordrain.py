from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import os
import glob
from gensim import utils
import gensim
from sklearn import preprocessing
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.pyplot import plot, show, bar, grid, axis, savefig, clf
import matplotlib.markers
import matplotlib.pyplot as plt
import matplotlib.font_manager as mfm
from mpl_toolkits import mplot3d
import random
import scipy
from scipy.sparse import csr_matrix
from matplotlib import cm
import unicodedata
import math
from matplotlib._enums import JoinStyle
from matplotlib.markers import MarkerStyle
import string
import hashlib
from bidi.algorithm import get_display
import json
from typing import NamedTuple


NR_OF_WORDS_TO_SHOW = 300
NGRAMS = (1,2)
FONTSIZE = 16
X_STRETCH = 30 # How much to make the x-coordinates wider
X_LENGTH = 10 # Default value for determine if two texts colide x-wise (typically changed for different corpora, then the default is not used)
Y_LENGTH = 10 # Determines if two texts colide y-wise, and how much to move, when the texts collide
DEBUG = False
ABOVE_BELOW_RATIO = 2
letters = list(string.ascii_lowercase) + ['å', 'ä', 'ö', 'ü', 'é', 'ß']
ALL_FOLDERS_NAME = "all"
NR_OF_WORDS_IN_BAR_PLOT = 60

#https://stackoverflow.com/questions/24581194/matplotlib-text-bounding-box-dimensions

#matplotlib.rcParams["font.family"] = "monospace"
#plt.rcParams["pdf.use14corefonts"] = True
#matplotlib.rc("text", usetex=True)

class Point(NamedTuple):
    x: float
    y: float
    
    def scalex(self, scale):
        return Point(self.x * scale, self.y)

class Rect(NamedTuple):
    min: Point
    max: Point
    
    def intersects(self, other):
        other_in_me_x = self.min.x <= other.min.x <= self.max.x
        me_in_other_x = other.min.x <= self.min.x <= other.max.x
        other_in_me_y = self.min.y <= other.min.y <= self.max.y
        me_in_other_y = other.min.y <= self.min.y <= other.max.y
        return (other_in_me_x or me_in_other_x) and (other_in_me_y or me_in_other_y)

def get_extra_indicator(idf, ngrams, output_folder):
    extra = "-n-grams-" + str(ngrams).replace(" ", "") + "-" + output_folder.replace("/", "-").replace("_", "-")
    if not idf:
        extra = extra + "-tf-only"
    else:
        extra = extra + "-tf-idf"
    return extra
        
        
def latex_clean(text):
    text = text.replace("_", " ")
    text = ''.join(ch for ch in text if ch.isalnum() or ch in [".", ",", " ", "!", "?", "-", "/"]).strip()
    return text
    
def construct_pdf_file_name(input_name, idf, ngrams, output_folder):
    return input_name.lower().replace(" ", "-").replace(":", "_").replace("å", "ao").replace("ö", "oo").replace("ä", "aa") + get_extra_indicator(idf, ngrams, output_folder) + ".pdf"
  
def word_in_terms_to_emphasize(word, terms_to_emphasize, mark_new_words, terms_to_emphasize_list_require_exact_match_ngrams, terms_to_emphasize_inverse):
    emphasize_term = word_in_terms_to_emphasize_before_inverse(word, terms_to_emphasize, mark_new_words, terms_to_emphasize_list_require_exact_match_ngrams)
    if terms_to_emphasize_inverse:
        return not emphasize_term
    else:
        return emphasize_term

def word_in_terms_to_emphasize_before_inverse(word, terms_to_emphasize, mark_new_words, terms_to_emphasize_list_require_exact_match_ngrams):

    if terms_to_emphasize == []:
        return False
                
    if terms_to_emphasize_list_require_exact_match_ngrams:
        if word in terms_to_emphasize or word.replace("_", " ") in terms_to_emphasize:
            return True
        else:
            return False
    else:
        if word in terms_to_emphasize:
            return True
        if mark_new_words: # TODO: Perhaps this functionality is interesting also otherwise
            for e in terms_to_emphasize:
                if e in word:
                    return True
        if terms_to_emphasize_list_require_exact_match_ngrams: #need exact match:
            for e in terms_to_emphasize:
                if e in word.replace("_", " ") or e in word.replace(" ", "_"):
                    return True
        if " " in word:
            splits = word.split(" ")
            for sp in splits:
                if sp in terms_to_emphasize:
                    return True
        if "_" in word:
            splits = word.split("_")
            for sp in splits:
                if sp in terms_to_emphasize:
                    return True
    return False
    
# Very simple compound spitting, not really language independent, but also not very well adapted to any language
def get_compound_vector(word, word2vec_model):
    return_vector = None
    for i in range(4, len(word) - 3):
        first = word[:i].lower()
        second = word[i:].lower()
        if first in word2vec_model and second in word2vec_model:
            #print("Found both second and first", word)
            #first_v = word2vec_model[first]
            second_v = word2vec_model[second]
            #return_vector = np.add(first_v, second_v)
            return second_v
                
    for j in range(len(word)-1, 4, -1):
        second_alone = word[-j:].lower()
        if second_alone in word2vec_model:
            return_vector = word2vec_model[second_alone]
            return return_vector
            
        first_alone = word[:j].lower()
        if first_alone in word2vec_model:
            return_vector = word2vec_model[first_alone]
            return return_vector

            
    return return_vector
        
def get_vector_from_model(word, model):
    if word in model:
        return model[word]
    if len(word) > 1:
        word_with_upper_init = word[0].upper() + word[1:]
        if word_with_upper_init in model:
            return model[word_with_upper_init]
            
    word = unicodedata.normalize("NFC", word)
    if word in model:
        return model[word]
    word = unicodedata.normalize("NFD", word)
    if word in model:
        return model[word]
    word = unicodedata.normalize("NFKC", word)
    if word in model:
        return model[word]
    if word.replace("ß", "ss") in model:
        return model[word.replace("ß", "ss")]
    if word.replace("ss", "ß") in model:
        return model[word.replace("ss", "ß")]
    return None
        
    
def get_vector_for_word(word, word2vec_model, ngram_range):
    vec_raw = get_vector_from_model(word, word2vec_model)
    if type(vec_raw) == np.ndarray: # Not None
        length = vec_raw.shape[0]
        norm_vector = list(preprocessing.normalize(np.reshape(vec_raw, newshape = (1, length)), norm='l2')[0])
        return norm_vector
    
    # try with lower_case_version
    vec_raw = get_vector_from_model(word.lower(), word2vec_model)
    if type(vec_raw) == np.ndarray: # Not None
        length = vec_raw.shape[0]
        norm_vector = list(preprocessing.normalize(np.reshape(vec_raw, newshape = (1, length)), norm='l2')[0])
        return norm_vector
        
    if " " in word or "_" in word:
        if " " in word:
            sub_words = word.split(" ")
        elif "_" in word:
            sub_words = word.split("_")
        acc_vector = None
        for sw in sub_words:
            vec_raw = get_vector_from_model(sw, word2vec_model)
            if type(vec_raw) == np.ndarray: # Not None
                if ngram_range[0] == 1 and "_" not in word:
                    acc_vector = vec_raw # This will lead to that only the last in a +1-gram
                # will contribute to the vector. But otherwise the tsne space puts to much
                # emphasis on the +1-grams
                else:
                    if type(acc_vector) == np.ndarray:
                        acc_vector = np.add(vec_raw, acc_vector)
                    else:
                        acc_vector = vec_raw
                
        if type(acc_vector) != np.ndarray:
            compound_vector = get_compound_vector(word, word2vec_model)
            if type(compound_vector) == np.ndarray:
                acc_vector = compound_vector
        if type(acc_vector) == np.ndarray: #Found compound vector
            length = acc_vector.shape[0]
            norm_vector = list(preprocessing.normalize(np.reshape(acc_vector, newshape = (1, length)), norm='l2')[0])
            return norm_vector
            
    compound_vector = get_compound_vector(word, word2vec_model)
    if type(compound_vector) == np.ndarray:
        length = compound_vector.shape[0]
        norm_vector = list(preprocessing.normalize(np.reshape(compound_vector, newshape = (1, length)), norm='l2')[0])
        return norm_vector
    
    for i in range(0, len(word)):
        for l in letters:
            edited_word = word[:i] + l + word[i+1:]
            if edited_word in word2vec_model:
                #print("found", word, edited_word)
                vec_raw = word2vec_model[edited_word]
                length = vec_raw.shape[0]
                norm_vector = list(preprocessing.normalize(np.reshape(vec_raw, newshape = (1, length)), norm='l2')[0])
                return norm_vector
                
    print("No word2vec vector found for:", word)
    return None


def do_plot(fig, word_vec_dict, sorted_word_scores_vec, title, idf, ngram_range,  x_length_factor, times_found_dict, nr_of_texts, max_score, min_score, new_words, extra, y_length_factor, fontsize, same_y_axis, plot_nr, mark_new_words, bar_height_boost, bar_strength_boost, bar_start_boost, extreme_values_tuple=None, add_title=False, terms_to_emphasize=[], log_for_bar_height=False, terms_to_emphasize_list_require_exact_match_ngrams=False, terms_to_emphasize_inverse=False, scale_max_x_used=1.3, freq_dict=None, min_f_in_current_document=1, show_number_of_occurrences=False, plot_mushroom_head=True, actual_plotted_word_size_dict=None, y_extra_space_factor=1.0, y_extra_pow_factor=0, above_below_ratio=ABOVE_BELOW_RATIO, show_words_outside_ylim = False, nr_of_vertical_lines = 0):
    
    emphasis_color = (0.45, 0.45, 0.45)
    emphasis_color_down = (0.7, 0.7, 0.7)
    
    print("Do plot for: ", title)
    if extreme_values_tuple:
        if DEBUG:
            extreme_marker_color = "red"
        else:
            extreme_marker_color = (0.0, 0.0, 0.0, 0.0)
            
        global_point_x_overall_min, global_point_x_overall_max, \
                global_point_y_overall_min, global_point_y_overall_max, global_point_x_overall_max_with_text = extreme_values_tuple
        
    point_x_overall_min = math.inf
    point_x_overall_max = -math.inf
    point_x_overall_max_with_text = -math.inf
    point_y_overall_min = math.inf
    point_y_overall_max = -math.inf

    cmap = plt.get_cmap('cool')
    plt.axis('off')
    
    # To avoid auto-scaling, set limits for the axes
    margin = 1.1
    right_margin = 1.3
    all_x = [x*X_STRETCH for x, y in word_vec_dict.values()]
    axes = plt.gca()
    ymin_for_ylim_without_margin = -max_score*above_below_ratio
    ymin_for_ylim = ymin_for_ylim_without_margin*margin
    xmin_for_xlim = min(all_x)*margin
    xmax_for_xlim = max(all_x)*right_margin*scale_max_x_used
    axes.set_xlim(xmin=xmin_for_xlim, xmax=xmax_for_xlim)
    axes.set_ylim(ymin=ymin_for_ylim, ymax=max_score*margin)
    
    vertical_line_position = xmin_for_xlim
    vertical_line_width = (xmax_for_xlim - xmin_for_xlim)/10000
    vertical_line_style = 'dotted'
    for vertical_line_nr in range(0, nr_of_vertical_lines):
        plt.vlines(vertical_line_position, ymin_for_ylim, max_score*margin, colors=None, linestyles=vertical_line_style, zorder = -30000000, linewidth=vertical_line_width, color='lightgrey')
        plt.text(vertical_line_position, max_score*margin, str(vertical_line_nr), zorder = -30000000, fontsize=2)
        plt.text(vertical_line_position, ymin_for_ylim, str(vertical_line_nr), zorder = -30000000, fontsize=2)
        vertical_line_position = vertical_line_position +  (xmax_for_xlim-xmin_for_xlim)/nr_of_vertical_lines
        if vertical_line_style == 'dotted':
            vertical_line_style = 'dashdot'
        else:
            vertical_line_style = 'dotted'
        
    texts = []
    z_order_text = 0
    word_nr = 1
    #fs = FONTSIZE
        
    previous_points = []
    used_colors_list = []
    actual_plotted_word_size_dict_to_create = {}
    for score, word in sorted_word_scores_vec:
        if freq_dict[word] < min_f_in_current_document: # To infrequent to include in plot
            continue
            
       
    
        text_color = (0.0, 0.0, 0.0)
                    
        fontsize_base = fontsize
        if fontsize > max_score:
            fontsize_base = max_score
 
                
        fs = fontsize_base*score/max_score
        fontsize_to_use = fs
        if fontsize_to_use < 1:
            fontsize_to_use = 1 # if it goes below 1, it is 1 anyway.
            if DEBUG:
                text_color = 'red'
        
        y_extra_space_factor_to_use = y_extra_space_factor * round(math.pow(fontsize_to_use, y_extra_pow_factor), 3)
        if y_extra_space_factor_to_use < 1:
            y_extra_space_factor_to_use = 1
            
        # If the word is underlined, a little extra space is needed
        if word_in_terms_to_emphasize(word, terms_to_emphasize, mark_new_words, terms_to_emphasize_list_require_exact_match_ngrams, terms_to_emphasize_inverse):
            y_extra_space_factor_to_use = y_extra_space_factor_to_use*1.1
            
        if word not in word_vec_dict:
            print("unknown word: ", word)
            exit(1)
        if word in word_vec_dict:
            word_to_display = word.replace(" ", "_")
            if show_number_of_occurrences and freq_dict and word in freq_dict:
                if not(isinstance(show_number_of_occurrences, int)):
                    show_number_of_occurrences = 10 # Backward compatibility, used to be a boolean only
                if freq_dict[word] <= show_number_of_occurrences:
                    word_to_display = word_to_display + "(" + str(int(freq_dict[word])) + ")"
            
            word_to_display = get_display(word_to_display)
            len_of_word = len(word_to_display)
            
            fontstyle = "normal"
            fontweight = "normal"
            nr_of_texts_except_all = nr_of_texts - 1
            if title != ALL_FOLDERS_NAME:
                if word in times_found_dict and times_found_dict[word] != nr_of_texts_except_all or nr_of_texts_except_all <= 1:
                    fontstyle = "normal"
                else:
                    fontstyle = "italic"
                        
                if mark_new_words:
                    if plot_nr != 0 and word in new_words:
                        # First time it appears
                        fontweight = "bold"
                        word_to_display = word_to_display + "*"
                    if word in times_found_dict and times_found_dict[word] == nr_of_texts_except_all:
                        fontstyle = "italic" #found in all texts
                        fontweight = "light"
                else:
                    if word in times_found_dict and times_found_dict[word] == 1 and nr_of_texts_except_all > 1: # only occurs in this text
                        #print("found once, bold", word)
                        fontweight = "bold"
                        #fontweight = "light"
            
            
            point = Point(word_vec_dict[word][0], word_vec_dict[word][1])
            
            point = point.scalex(X_STRETCH)
            

            
            
            point_before = point
            changed = True
            
            # Just use a dummy-point the first round
            max_point_x = point.x
  
            # The second round, use the actual size
            if actual_plotted_word_size_dict is not None :
                assert(word in actual_plotted_word_size_dict)
                bbox_x = actual_plotted_word_size_dict[word][0]
                max_point_x = point.x + bbox_x
            
            while changed:
                point_y_before_loop = point.y
                for prev_word, prev_point in previous_points:
                        
                    # Just use a dummy-point the first round
                    left_upper_max_point_y = point.y
                    
                    # The second round, use the actual size
                    if actual_plotted_word_size_dict is not None :
                        assert(word in actual_plotted_word_size_dict)
                        bbox_y = actual_plotted_word_size_dict[word][1]
                        left_upper_max_point_y = point.y + bbox_y*y_extra_space_factor_to_use
                
                    my_bounding_box = Rect(point, Point(max_point_x, left_upper_max_point_y))
                    if my_bounding_box.intersects(prev_point):
                        point = Point(point.x, prev_point.max.y)
                if point_y_before_loop == point.y:
                    changed = False
   
    
            y_bar_start = (max_score - ymin_for_ylim)*0.01*bar_start_boost
           
            
            cmap_value = ((point.x/X_STRETCH)/100 + 1)/2 # Not used when actually plotting
            if extreme_values_tuple:
                cmap_value = (point.x - global_point_x_overall_min)/(global_point_x_overall_max - global_point_x_overall_min)
            color_lighter_value = min(0.15*fontsize_to_use, 1)
            color_lighter = (cmap(cmap_value)[0], cmap(cmap_value)[1], cmap(cmap_value)[2], color_lighter_value)
            color_lighter_lighter_value = min(0.1*fontsize_to_use, 1)
            color_lighter_lighter = (cmap(cmap_value)[0], cmap(cmap_value)[1], cmap(cmap_value)[2], color_lighter_lighter_value)
            
            marker=MarkerStyle('v', joinstyle=JoinStyle.miter)
            
            # The bar upwards
            y_bar_length = score
            linewidth = bar_strength_boost*score/max_score
            upwardbar_end = y_bar_start + y_bar_length
            color_of_upward_bar = cmap(cmap_value)
            used_colors_list.append(color_of_upward_bar)
            
            # Make lines for terms to emphasize in black
            selected_lexicon_term = False
            if word_in_terms_to_emphasize(word, terms_to_emphasize, mark_new_words, terms_to_emphasize_list_require_exact_match_ngrams, terms_to_emphasize_inverse):
                fontweight = "bold"
                selected_lexicon_term = True
                
                used_colors_list[-1] = emphasis_color
                plt.plot([point.x, point.x], [y_bar_start, upwardbar_end], color=emphasis_color, zorder = -z_order_text, linewidth=linewidth)
                if plot_mushroom_head:
                    plt.scatter(point.x, upwardbar_end, zorder = -z_order_text, color=emphasis_color, marker = marker, s=linewidth) #
                    
            elif mark_new_words and plot_nr != 0 and word in new_words:
                used_colors_list[-1] = emphasis_color
                plt.plot([point.x, point.x], [y_bar_start, upwardbar_end], color=emphasis_color, zorder = -z_order_text, linewidth=linewidth)
                if plot_mushroom_head:
                    plt.scatter(point.x, upwardbar_end, zorder = -z_order_text, color=emphasis_color, marker = "*", s=linewidth)
            else:
                plt.plot([point.x, point.x], [y_bar_start, upwardbar_end], color=color_of_upward_bar, zorder = -2000-z_order_text, linewidth=linewidth)
                if plot_mushroom_head:
                    plt.scatter(point.x, upwardbar_end, zorder = -200000, color=color_of_upward_bar, marker = "o", s=linewidth) #s=
                
                
            
            # The texts have been changed to be printed downwards instead
            negative_text_y = -point.y
                        
            
            if show_words_outside_ylim or negative_text_y >= ymin_for_ylim_without_margin:
                # Mark lexicon terms
                if selected_lexicon_term:
                    t = plt.text(point.x, negative_text_y, word_to_display, zorder = z_order_text, color = text_color, fontsize=fontsize_to_use, fontstyle=fontstyle, fontweight="bold", verticalalignment="top")
                                    
                    t = plt.text(point.x, negative_text_y, len(word)*"_", zorder = -20000, color = (0.0, 0.0, 0.0), fontsize=fontsize_to_use, fontstyle="italic", fontweight="light", verticalalignment="top")
                else:
                    t = plt.text(point.x, negative_text_y, word_to_display, zorder = z_order_text, color = text_color, fontsize=fontsize_to_use, fontstyle=fontstyle, fontweight=fontweight, verticalalignment="top")
            else:
                plt.scatter(point.x, ymin_for_ylim_without_margin, zorder = -200000, color=color_lighter_lighter, marker = ".", s=fontsize_to_use*0.01)
                negative_text_y = ymin_for_ylim_without_margin # to make the bar downward stop
                if plot_nr != 0 and word in new_words:
                    # Add new words at the border, but don't make them bold (just for estetical reasons)
                    t = plt.text(point.x, ymin_for_ylim_without_margin*1.01, word_to_display, zorder = z_order_text, color = emphasis_color, fontsize=fontsize_to_use, fontstyle=fontstyle, fontweight="normal", verticalalignment="top", rotation=270)
                else:
                # For all other words, add a dummy-text and a marker
                    t = plt.text(point.x, ymin_for_ylim_without_margin, "", zorder = z_order_text, color = text_color, fontsize=fontsize_to_use, fontstyle=fontstyle, fontweight=fontweight, verticalalignment="top")
                    
            
            bbox = t.get_window_extent(renderer = fig._get_renderer())
            transf = plt.gca().transData.inverted()
            bb_datacoords = bbox.transformed(transf)
            x_length_from_bbox = bb_datacoords.x1 - bb_datacoords.x0
            y_length_from_bbox = bb_datacoords.y1 - bb_datacoords.y0
            
      
            max_point_x = point.x + x_length_from_bbox
            max_point_y = point.y + y_length_from_bbox*y_extra_space_factor_to_use
               
                
            max_point = Point(max_point_x, max_point_y)
            previous_points.append((word_to_display, Rect(point, max_point)))
            
            actual_plotted_word_size_dict_to_create[word] = (x_length_from_bbox, y_length_from_bbox)
            
            
            # The bar downwards
            linewidth_down=bar_strength_boost*score/max_score/fs
            
            if word_in_terms_to_emphasize(word, terms_to_emphasize, mark_new_words, terms_to_emphasize_list_require_exact_match_ngrams, terms_to_emphasize_inverse):
                plt.plot([point.x, point.x], [y_bar_start, negative_text_y], color=emphasis_color_down, linewidth=linewidth_down, zorder = -10000000)
                plt.scatter(point.x, negative_text_y, zorder = -100000, color=emphasis_color_down, marker = marker, s=fontsize_to_use*0.01) #
            elif mark_new_words and plot_nr != 0 and word in new_words:
                used_colors_list[-1] = emphasis_color_down
                plt.plot([point.x, point.x], [y_bar_start, negative_text_y], color=emphasis_color_down, linewidth=linewidth_down, zorder = -30000000)
                plt.scatter(point.x, negative_text_y, zorder = -300000, color=emphasis_color_down, marker = marker, s=fontsize_to_use*0.01) #
            else:
                plt.plot([point.x, point.x], [y_bar_start, negative_text_y], color=color_lighter, linewidth=linewidth_down, zorder = -20000000)
                plt.scatter(point.x, negative_text_y, zorder = -200000, color=color_lighter_lighter, marker = "o", s=fontsize_to_use*0.01) #

            
            z_order_text =  z_order_text - 1
   
            min_y_used = point.y
            max_y_used = upwardbar_end + fontsize_to_use
            if min_y_used < point_y_overall_min:
                point_y_overall_min = min_y_used
            if max_y_used > point_y_overall_max:
                point_y_overall_max = max_y_used
                
            max_x_used_with_text = max_point_x*scale_max_x_used
            
            if point.x < point_x_overall_min:
                point_x_overall_min = point.x
            if point.x > point_x_overall_max:
                point_x_overall_max = point.x
            if max_x_used_with_text > point_x_overall_max_with_text:
                point_x_overall_max_with_text = max_x_used_with_text
            
        else:
            print("Word not found, strange", word)
        word_nr = word_nr + 1

    x_width =  point_x_overall_max - point_x_overall_min
    title_to_use = "\n" + title
    title_to_use = latex_clean(title_to_use)
    title_fontsize = 10/math.log(len(title_to_use) + 3)*1.6
    if add_title:
        plt.title(title_to_use, fontsize=title_fontsize)
    
    return (point_x_overall_min, point_x_overall_max, point_y_overall_min, point_y_overall_max, point_x_overall_max_with_text, used_colors_list, actual_plotted_word_size_dict_to_create)


def vectorize_and_generate_plotdata(background_corpus, texts, names, stopwords, word2vec_model, ngram_range, nr_of_words_to_show, output_folder, x_length_factor, idf, extra_words, y_length_factor, fontsize, same_y_axis, mark_new_words, bar_height_boost, bar_strength_boost, bar_start_boost, min_df, max_df, add_title, terms_to_emphasize_list, log_for_bar_height, min_df_in_document, terms_to_emphasize_list_require_exact_match_ngrams, terms_to_emphasize_inverse, scale_max_x_used, min_f_in_current_document, show_number_of_occurrences, plot_mushroom_head, min_tf_in_corpora, lowercase, include_all_terms_to_emphasize, y_extra_space_factor, y_extra_pow_factor, above_below_ratio, use_global_max_score, show_words_outside_ylim, nr_of_vertical_lines):

    
    if terms_to_emphasize_list:
        if lowercase:
            lowered_terms_to_emphasize_list = []
            for sublist in terms_to_emphasize_list:
                lowered_terms_to_emphasize_list.append([term.lower() for term in sublist])
            terms_to_emphasize_list = lowered_terms_to_emphasize_list
            
    
    # Create folder for storing tf-idf or tf statistics
    extra = get_extra_indicator(idf, ngram_range, output_folder)
    texts_folder = "texts" + get_extra_indicator(idf, ngram_range, output_folder)
    texts_output = os.path.join(output_folder, texts_folder)
    if not os.path.exists(texts_output):
        os.mkdir(texts_output)
    
    nr_of_texts = len(texts)
    
    # For tf-idf
    ############
        
    freq_dict_list = [] # A list of word freqencies-dictionaries for each document
    freqs_for_entire_corpora = {}
    min_freq_in_a_doc_for_each_word = {}
    
    for text in texts:
        # A local vectorizer for just each text, to count frequencies in each text and save in freq_dict_list
        freq_vectorizer = TfidfVectorizer(sublinear_tf=False, ngram_range = ngram_range, use_idf=False, norm=None, stop_words=stopwords, lowercase=lowercase)
        freq_X = freq_vectorizer.fit_transform([text])
        freqency_inversed = freq_vectorizer.inverse_transform(freq_X)
        freq_list = []
        freq_dict = {}
        for words, freqs in zip(freqency_inversed, freq_X):
            score_vec = freqs.toarray()[0]
            for word in words:
                freq_for_word = score_vec[freq_vectorizer.vocabulary_[word]]
                freq_dict[word] = freq_for_word
                
                # Count the total frequency for a word, in all texts
                if word not in freqs_for_entire_corpora:
                    freqs_for_entire_corpora[word] = freq_for_word
                else:
                    freqs_for_entire_corpora[word] =  freqs_for_entire_corpora[word] + freq_for_word
                    
                # Collect minimum occurrences in the corpus for a word in a document
                if word not in min_freq_in_a_doc_for_each_word:
                    min_freq_in_a_doc_for_each_word[word] = freq_for_word
                else:
                    if freq_for_word < min_freq_in_a_doc_for_each_word[word]:
                        min_freq_in_a_doc_for_each_word[word] = freq_for_word
                
        freq_dict_list.append(freq_dict)

    # Create a vocabulary where all words with a frequencey < min_tf_in_corpora are excluded
    vocabulary_for_min_tf_in_corpora = set([word for (word, freq) in freqs_for_entire_corpora.items() if freq >= min_tf_in_corpora])
    
    if min_df_in_document != None:
        print("WARNING: min_df_in_document is used, not necessarily practical")
        vocabulary_min_df_in_document = set([word for (word, freq) in min_freq_in_a_doc_for_each_word.items() if freq >= min_df_in_document])
        vocabulary_to_use = list(vocabulary_for_min_tf_in_corpora.intersection(vocabulary_min_df_in_document))
    else:
        vocabulary_to_use = list(vocabulary_for_min_tf_in_corpora)
    
    if max_df != 1.0 or min_df > 1:
        max_df_vectorizer = TfidfVectorizer(stop_words=stopwords, min_df=min_df, max_df=max_df, smooth_idf=False, sublinear_tf=False, ngram_range = ngram_range, use_idf=idf, norm=None, lowercase=lowercase)
        print("before, max_df, min_df",  len(vocabulary_to_use))
        print("nr of texts", len(texts))
        max_df_X = max_df_vectorizer.fit_transform(texts)
        max_df_set = set(max_df_vectorizer.vocabulary_)
        vocabulary_to_use = max_df_set.intersection(set(vocabulary_to_use))
        print("after, max_df, min_df", len(vocabulary_to_use))
        
    print("before",  len(vocabulary_to_use))
    if include_all_terms_to_emphasize:
        flattened_terms_to_emphasize_list = []
        for sublist in terms_to_emphasize_list:
            flattened_terms_to_emphasize_list.extend(sublist)
        vocabulary_to_use = list(set(vocabulary_to_use).union(set(flattened_terms_to_emphasize_list)))
    print("after", len(vocabulary_to_use))
    # The main vectorizer, with the statistics to use for plotting
    main_vectorizer = TfidfVectorizer(stop_words=stopwords, min_df=min_df, max_df=max_df, smooth_idf=False, sublinear_tf=False, ngram_range = ngram_range, use_idf=idf, norm=None, vocabulary = vocabulary_to_use, lowercase=lowercase)

    print("len(texts), all texts inclusive", len(texts))
    print("len(texts + background_corpus)", len(texts + background_corpus))

    main_vectorizer.fit_transform(texts + background_corpus)
    X = main_vectorizer.transform(texts)
    inversed = main_vectorizer.inverse_transform(X)
    
    sorted_word_scores_matrix_before_cutoff = []
    all_words = []
    for el, w, name in zip(inversed, X, names):
        score_vec = w.toarray()[0]
        word_scord_vec = []
        for word in el:
            all_words.append(word)
            word_scord_vec.append((score_vec[main_vectorizer.vocabulary_[word]], word))
        word_scord_vec = sorted(word_scord_vec, reverse=True)
        sorted_word_scores_matrix_before_cutoff.append(word_scord_vec)
         
    
    # Create space (and score statistics):
    
    times_found_dict = {}
    max_scores = []
    min_scores = []
        
    # Only plot top nr_of_words_to_show words
    # And also, filter out too infrequent words
    new_words_lists_before_cutoff = []
    previous_words_set_before_cutoff = set()
    
    if nr_of_words_to_show is None:
        nr_of_words_to_show = math.inf
        print("nr_of_words_to_show is None, therefore it will be set to math.inf ")
                
    sorted_word_scores_matrix = []
    for sws, freq_dict in zip(sorted_word_scores_matrix_before_cutoff, freq_dict_list):
        filtered_sws_for_current_doc_freq = []
        
        new_words_list_before_cutoff = []
                
        for sw in sws:
            w = sw[1]
            if len(filtered_sws_for_current_doc_freq) < nr_of_words_to_show:
                # cut-off, not more than nr_of_words_to_show
                if freq_dict[w] < min_f_in_current_document:
                    pass # Too infrequent in document to include in plot
                else:
                    filtered_sws_for_current_doc_freq.append(sw)
                    
            # For checking if a word is one not in previous (before cut-off) corpora
            if w not in previous_words_set_before_cutoff:
                new_words_list_before_cutoff.append(w)
            previous_words_set_before_cutoff.add(w)
            
        sorted_word_scores_matrix.append(filtered_sws_for_current_doc_freq)
        new_words_lists_before_cutoff.append(new_words_list_before_cutoff)
    
    
    word_to_use_set = set()
    new_words_lists = []
    previous_words_set = set()
    max_all_plot = -math.inf
    for sws, title, freq_dict in zip(sorted_word_scores_matrix, names, freq_dict_list):
    
        max_score_subcorpus = -math.inf
        min_score_subcorpus = math.inf
        new_words_list = []
        
        for s, w in sws:
            assert(not(freq_dict[w] < min_f_in_current_document))
            # should have been removed in previous step
            
            word_to_use_set.add(w)
            if title != ALL_FOLDERS_NAME:  # Don't count the 'all image' as a document occurrences
                if w not in times_found_dict:
                    times_found_dict[w] = 1
                else:
                    times_found_dict[w] = times_found_dict[w] + 1
            
                # "All" has very different scores
                if s > max_score_subcorpus:
                    max_score_subcorpus = s
                if s < min_score_subcorpus:
                    min_score_subcorpus = s
            
            else:
                if s > max_all_plot:
                    max_all_plot = s
            # For checking if a word is one not in previous (cut-off-filtered) corpora
            if w not in previous_words_set:
                new_words_list.append(w)
            previous_words_set.add(w)
         

    
        new_words_lists.append(new_words_list)
        max_scores.append(max_score_subcorpus)
        min_scores.append(min_score_subcorpus)

    all_vectors_list = []
    found_words = []
    not_found_words = []

    
    # It is possible to add extra words, not found in the corpus, but that are to
    # be part of the visualisaion
    words_to_generate_space_for = word_to_use_set
    for word in extra_words:
        words_to_generate_space_for.add(word)
    words_to_generate_space_for = sorted(list(words_to_generate_space_for))
    print("Nr of words to generate space for:", len(words_to_generate_space_for))
    
    for word in words_to_generate_space_for:
        # Add vectors
        norm_vector = get_vector_for_word(word, word2vec_model, ngram_range)
        if norm_vector == None:
            not_found_words.append(word)
        else:
            found_words.append(word)
            all_vectors_list.append(norm_vector)
                    
    # Just add a random point for words not found in the space
    for word in not_found_words:
        print("Use random point for ", word)
        #seed = hash(word.encode('utf-8'))
        #print(seed)
        random_el = (random.sample(all_vectors_list, 1))[0]
        point = [el + 0.01*random.randint(1,10) for el in random_el]
        norm_vector = list(preprocessing.normalize(np.reshape(point, newshape = (1, len(point))), norm='l2')[0])
        found_words.append(word)
        all_vectors_list.append(norm_vector)
        
    all_vectors_np = np.array(all_vectors_list)
    print("len(all_vectors_list)", len(all_vectors_list))
    pca_model = PCA(n_components=50, random_state=0)
    tsne_model = TSNE(n_components=1, random_state=0)
    DX_pca = pca_model.fit_transform(all_vectors_np)
    DX = tsne_model.fit_transform(DX_pca)

    
    word_vec_dict = {}
    min_x = 100
    min_y = 100
    max_x = -100
    max_x = -100
    max_y = -100
    for point, found_word in zip(DX, found_words):
        point = [float(point[0]), 0]
        word_vec_dict[found_word] = point
        if point[0] < min_x:
            min_x = point[0]
        if point[0] > max_x:
            max_x = point[0]
        if point[1] < min_y:
            min_y = point[1]
        if point[1] > max_y:
            max_y = point[1]


    # Write scores to file
    for sws, title, freq_dict in zip(sorted_word_scores_matrix, names, freq_dict_list):
        tf_name = os.path.join(texts_output, title + extra + ".txt")
        with open(tf_name, "w") as tf_write:
            for s, w in sws:
                w_freq = freq_dict[w]
                w_vec = word_vec_dict[w]
                tf_write.write("\t".join([w, str(s), "(" + str(w_freq) + ")", str(w_vec)]) + "\n")


        
    global_point_x_overall_min = math.inf
    global_point_x_overall_max = -math.inf
    global_point_x_overall_max_with_text = -math.inf
    global_point_y_overall_min = math.inf
    global_point_y_overall_max = -math.inf
    
    actual_plotted_word_size_dict_list = []
    
    max_score_except_all = max(max_scores)
    
    # Make one first run, just to get the global max and min point (but don't save the result)
    for inverse, sorted_word_scores, title, min_score, new_words, plot_nr, terms_to_emphasize, freq_dict in zip(inversed, sorted_word_scores_matrix, names, min_scores, new_words_lists, range(0, len(names)), terms_to_emphasize_list, freq_dict_list):
    
        max_score_to_use_in_round = max_score_except_all
        
         
        # Either use global or visualisation-local max_score, except for "all".
        if title == ALL_FOLDERS_NAME:
            max_score_to_use_in_round = max_all_plot
        elif not use_global_max_score:
            max_score_to_use_in_round = max_scores[plot_nr]
            
        main_fig = plt.figure()
        plt.axis('off')
        point_x_overall_min, point_x_overall_max, point_y_overall_min, point_y_overall_max, point_x_overall_max_with_text, used_colors_list, actual_plotted_word_size_dict = \
            do_plot(main_fig, word_vec_dict, sorted_word_scores, title, idf, ngram_range, x_length_factor, times_found_dict, nr_of_texts, max_score_to_use_in_round, min_score, new_words, extra=extra, fontsize=fontsize, y_length_factor=y_length_factor, plot_nr = plot_nr, same_y_axis = same_y_axis, mark_new_words=mark_new_words, bar_height_boost=bar_height_boost, bar_strength_boost=bar_strength_boost, bar_start_boost=bar_start_boost, add_title=add_title, terms_to_emphasize=terms_to_emphasize, log_for_bar_height=log_for_bar_height, terms_to_emphasize_list_require_exact_match_ngrams=terms_to_emphasize_list_require_exact_match_ngrams, terms_to_emphasize_inverse=terms_to_emphasize_inverse, scale_max_x_used=scale_max_x_used, freq_dict=freq_dict, min_f_in_current_document=min_f_in_current_document, show_number_of_occurrences=show_number_of_occurrences, plot_mushroom_head=plot_mushroom_head, y_extra_space_factor=y_extra_space_factor, y_extra_pow_factor=y_extra_pow_factor, above_below_ratio=above_below_ratio, show_words_outside_ylim = show_words_outside_ylim, nr_of_vertical_lines = nr_of_vertical_lines)
         
        actual_plotted_word_size_dict_list.append(actual_plotted_word_size_dict)
        
        # TODO: These are not used anymore
        if point_x_overall_min < global_point_x_overall_min:
            global_point_x_overall_min = point_x_overall_min
        if point_x_overall_max > global_point_x_overall_max:
            global_point_x_overall_max = point_x_overall_max
        if point_x_overall_max_with_text > global_point_x_overall_max_with_text:
            global_point_x_overall_max_with_text = point_x_overall_max_with_text
        if point_y_overall_min < global_point_y_overall_min:
            global_point_y_overall_min = point_y_overall_min
        if point_y_overall_max > global_point_y_overall_max:
            global_point_y_overall_max = point_y_overall_max
          
        main_fig.clf()
        plt.close('all')
            
    extreme_values_tuple = (global_point_x_overall_min, global_point_x_overall_max, \
                global_point_y_overall_min, global_point_y_overall_max, global_point_x_overall_max_with_text)
                
    # Make one another run and save the results in images
    for inverse, sorted_word_scores, title, new_words, plot_nr, terms_to_emphasize, freq_dict, actual_plotted_word_size_dict in zip(inversed, sorted_word_scores_matrix, names,  new_words_lists, range(0, len(names)), terms_to_emphasize_list, freq_dict_list, actual_plotted_word_size_dict_list):
    
        # Either use global or visualisation-local max_score, except for "all".
        max_score_to_use_in_round = max_score_except_all
        if title == ALL_FOLDERS_NAME:
            max_score_to_use_in_round = max_all_plot
        elif not use_global_max_score:
            max_score_to_use_in_round = max_scores[plot_nr]
            
        main_fig = plt.figure()
        plt.axis('off')
        do_plot_args = [
        word_vec_dict,
        sorted_word_scores, title, idf, ngram_range,
        x_length_factor, times_found_dict, nr_of_texts,
        max_score_to_use_in_round, min_score, new_words
        ]
        do_plot_kwargs = {}
        do_plot_kwargs.update(extra = extra, fontsize=fontsize, y_length_factor=y_length_factor, plot_nr = plot_nr, same_y_axis = same_y_axis, mark_new_words=mark_new_words, bar_height_boost=bar_height_boost, bar_strength_boost=bar_strength_boost, bar_start_boost=bar_start_boost, extreme_values_tuple = extreme_values_tuple, add_title=add_title, terms_to_emphasize=terms_to_emphasize, log_for_bar_height=log_for_bar_height, terms_to_emphasize_list_require_exact_match_ngrams=terms_to_emphasize_list_require_exact_match_ngrams, terms_to_emphasize_inverse=terms_to_emphasize_inverse, scale_max_x_used=scale_max_x_used, freq_dict=freq_dict, min_f_in_current_document=min_f_in_current_document, show_number_of_occurrences=show_number_of_occurrences, plot_mushroom_head=plot_mushroom_head, actual_plotted_word_size_dict=actual_plotted_word_size_dict, y_extra_space_factor=y_extra_space_factor, y_extra_pow_factor=y_extra_pow_factor, above_below_ratio=above_below_ratio, show_words_outside_ylim=show_words_outside_ylim, nr_of_vertical_lines = nr_of_vertical_lines)
        file_name = os.path.join(output_folder, construct_pdf_file_name(title, idf, ngram_range, output_folder))
        with open(file_name + ".json", "wt") as json_file:
            json.dump({"args":do_plot_args,"kwargs":do_plot_kwargs}, json_file)
            
        main_fig.clf()
        plt.close('all')

"""
def plot_round(json_data):
        main_fig = plt.figure()
        plt.axis('off')
        do_plot_json = json.loads(json_data)
        point_x_overall_min, point_x_overall_max, point_y_overall_min, point_y_overall_max, point_x_overall_max_with_text, used_colors_list, actual_plotted_word_size_dict = \
        do_plot(main_fig, *do_plot_json["args"], **do_plot_json["kwargs"])
        do_plot_json["kwargs"].update(actual_plotted_word_size_dict=actual_plotted_word_size_dict)
        main_fig.clf()
        plt.close('all')
        return json.dumps(do_plot_json)
"""

def plot_from_plotdata(names, idf, ngram_range, output_folder, json_datas, nr_of_words_in_bar_plot):
    for title, json_data in zip(names, json_datas):
        file_name = os.path.join(output_folder, construct_pdf_file_name(title, idf, ngram_range, output_folder))
        main_fig = plt.figure()
        plt.axis('off')
        do_plot_json = json.loads(json_data)
        
        
        do_plot(main_fig, *do_plot_json["args"], **do_plot_json["kwargs"])
 
        #plt.tight_layout()
        #orientation = "landscape",
        plt.savefig(file_name,  format="pdf", bbox_inches='tight', pad_inches=0)
        print("Saved plot in " + file_name)
        plt.close('all')
        
        sorted_word_scores = do_plot_json["args"][1]
        freq_dict = do_plot_json["kwargs"]["freq_dict"]
        min_f_in_current_document = do_plot_json["kwargs"]["min_f_in_current_document"]
        add_title = do_plot_json["kwargs"]["add_title"]
        # Figure with only frequencies
        bar_fig = plt.figure()
        plt.rcParams['ytick.labelsize']= 240/nr_of_words_in_bar_plot#4
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.tick_params(axis='both', which='both', length=0)
        ax.grid(False)
        ax.spines[['right', 'top', 'bottom', 'left']].set_visible(False)
        bar_file_name = file_name.replace(".pdf", "_bars.pdf")
        most_common_freq_filtered = [(x,y) for (x,y) in sorted_word_scores if freq_dict[y] >= min_f_in_current_document]
        most_common = most_common_freq_filtered[:nr_of_words_in_bar_plot]
        
        freqs = [round(x, 1) for (x,y) in most_common]
        words = []
        for (x, y) in most_common:
            word = y + " " + str(round(x,1))
            if y in freq_dict:
                word = word + "(" + str(int(freq_dict[y])) + ")"
            words.append(word)
            
        assert(len(freqs) == len(words))
        
        ax.barh(words, freqs, height=24/nr_of_words_in_bar_plot)
        ax.invert_yaxis()
        ax.invert_xaxis()
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        title_to_use = "\n" + title
        title_to_use = latex_clean(title_to_use)
        title_fontsize = 10/math.log(len(title_to_use) + 3)*1.6
        if add_title:
            plt.title(title_to_use, fontsize=title_fontsize)
        
        plt.savefig(bar_file_name, orientation = "landscape", format="pdf", bbox_inches='tight', pad_inches=0)
        
        
        print("Saved plot in " + bar_file_name)
        plt.close('all')

####
# The main function for generating the word rain
####
#
# scale_max_x_used: the extra space to the right, to make sure all words are in the picture
#
# Take care with the following: 'min_df_in_document' is a global cut-off. It removes words that occur less
# than 'min_df_in_document' in any document to be plotted anywhere. It is not that practical, and is only left for
# backward compatibility
#
# min_tf_in_corpora is also a global cut-off.
# it removes words that occur less than min_tf_in_corpora in the entire corpus to be plotted anywhere.
# Can for instance be matched with the same cut-off when creating a word2vec-model, to make sure that all words have a matching vector
#
#
# min_f_in_current_document is a local cut-off. It excludes words to be plotted that only occur rarely in the document that is plotted
#
# The font size will never be smaller than 1. Therefore, if you use to small font size for the largest words, the word rain will get boring, with most words in font size 1.
# include_all_terms_to_emphasize is default False. When true, all words in all_terms_to_emphasize are plotted (if included in the corpus), regardless of their min-tf and max-tf etc. (but the cut-off of top n, still holds though. So if their prominence is lower than other words, they willl still not be included)
# above_below_ratio: Decides how large part of the image will be above 0 (as bars) and how much will be below (words). Default is 2, i.e. double the space for the words compared to the bars.
# show_number_of_occurrences: False to don't show number of occurrences in paranthesis. Otherwise, the maximum number of occurrences, for which to show the the occurrences in paranthesis
# nr_of_words_to_show: The top nr_of_words_to_show. If None, it is set to math.inf. Don't do that if you don't have a very small data set though, because it will be take a very, very long time.
# nr_of_vertical_lines: How many vertical lines to print. Default is 0
#
# NOT VERY USEFUL. BUT KEPT, AS IT MIGHT GIVE A POSSIBILITY FOR FINE-TUNING
# These are not very useful. Might be kept a while, but will be removed later
# y_extra_pow_factor: the y-space that a word is allowed to take is factored by the (font size of that word)^0.5 as a default. By y_extra_pow_factor ^0.5 can be modified.
# y_extra_space_factor: also, the y-space that a word is allowed to take can be factored by y_extra_space_factor. This is 1 by default.
# log_for_bar_height is not used currently

# DEPRECATED, WILL BE REMOVED:
# bar_height_boost
# nr_iterations_for_fine_ajusting_the_placement: the plot algorithm is run several times, fine-justing the word positioning. The default is to run the fine-justing 1 time.
# same_y_axis is not used currently
# x_length_factor is not used, will be removed
# y_length_factor is (in the most recent version) only used as a factor for governing the height of the bars.
#

def generate_clouds(corpus_folder, word2vec_model, output_folder, ngrams=NGRAMS, nr_of_words_to_show = NR_OF_WORDS_TO_SHOW, x_length_factor = X_LENGTH, stopwords=[], background_corpus=[], pre_process_method = None, idf = True, extra_words_path = None, y_length_factor = Y_LENGTH, fontsize=FONTSIZE, same_y_axis=True, mark_new_words=False, min_df=1, max_df=1.0, add_title=True, bar_height_boost=1, bar_strength_boost=1, bar_start_boost = 1, terms_to_emphasize_list = [], log_for_bar_height=False, min_df_in_document=None, terms_to_emphasize_list_require_exact_match_ngrams=False, terms_to_emphasize_inverse=False, scale_max_x_used=1.3, slide_show=False, min_f_in_current_document=1, monospace=True, show_number_of_occurrences=False, nr_of_words_in_bar_plot=NR_OF_WORDS_IN_BAR_PLOT, plot_mushroom_head=True, pre_process_data=None, min_tf_in_corpora=1,  lowercase=True, include_all_terms_to_emphasize=False, y_extra_space_factor=1.0, y_extra_pow_factor=0, nr_iterations_for_fine_ajusting_the_placement=0, above_below_ratio = ABOVE_BELOW_RATIO, use_global_max_score=True, show_words_outside_ylim=False, nr_of_vertical_lines = 0):

    if monospace:
        matplotlib.rcParams["font.family"] = "monospace"
        
    # Since there is also one batch with all texts, which will add the
    # term frequency with one
    if isinstance(min_df, int):
        min_df = min_df + 1
    if isinstance(max_df, int):
        max_df = max_df + 1
        
    stopwords = list(stopwords)
    
    extra_words = []
    if extra_words_path:
        pn = os.path.join(extra_words_path, "*.txt")
        print("Extra words in", pn)
        files = glob.glob(pn)
        for f in files:
            with open(f) as opened:
                for line in opened:
                    extra_words.append(line.split("\t")[0])
   
    # Read texts that are to be visualised
    names = []
    texts = []
  
    folder_title = os.path.basename(corpus_folder)
    if folder_title.strip() == "":
        folder_title = os.path.basename(os.path.split(corpus_folder)[0])

    if not os.path.exists(corpus_folder):
        print("folder not found: ", corpus_folder)
        print("exit without plotting ", corpus_folder)
        exit()
    folders = glob.glob(os.path.join(corpus_folder, "*", ""))
    if len(folders) == 0:
        print("no subfolders found in ", corpus_folder)
        print("exit without plotting ", corpus_folder)
        exit()
    else:
        print(str(len(folders)) +  " nr of subfolders found in ", corpus_folder)
        

    folder_names = []
    for fn in folders:
        folder_name = unicodedata.normalize("NFC", os.path.basename(os.path.split(fn)[0]))
        folder_name_numeric = []
        
        for character in folder_name:
            if character.isnumeric():
                folder_name_numeric.append(character)
      
        if folder_name_numeric != []:
            folder_names.append((float("".join(folder_name_numeric)), folder_name, fn))
        else:
            folder_names.append((folder_name, folder_name, fn))
    
    folder_names.sort()
   
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
        
    tex_str = """
            \\documentclass{beamer}
            \\usepackage[utf8]{inputenc}
            """
    #if not slide_show:
    #    tex_str = tex_str + "\\usepackage[orientation=portrait,size=a4]{beamerposter}"
    tex_str = tex_str + \
            """
            \\title{
            """
    tex_str = tex_str + latex_clean(folder_title + " " + get_extra_indicator(idf, ngrams, output_folder))
    tex_str = tex_str + \
            """
            }
            \\author{}

            \\usepackage{graphicx}
            \\begin{document}

            \\maketitle
                    
            """
    bar_tex_str = tex_str
    texts_in_all_folders = []
    for number_name, folder_name, folder in folder_names:
        print("Reading files from: ", folder)
        #folder_name = unicodedata.normalize("NFC", os.path.basename(os.path.split(folder)[0]))
        texts_in_folder = []
        extract_files = glob.glob(os.path.join(folder, "*.txt"))
        print("Nr of found files: ", len(extract_files))
        if len(extract_files) == 0:
            print("no files to visualise found in ", folder)
            exit()
        for f in extract_files:
            with open(f) as openfile:
                text_in_file = openfile.read()
                if pre_process_method:
                    if pre_process_data:
                        text_in_file = pre_process_method(text_in_file, pre_process_data)
                    else:
                        text_in_file = pre_process_method(text_in_file)
                texts_in_folder.append(text_in_file)
                texts_in_all_folders.append(text_in_file)
        all_text_in_folder = "\n".join(texts_in_folder)
        
        texts.append(all_text_in_folder) # all files in folder to one text
        names.append(folder_name)
        
        
        
        pdf_file_name  = construct_pdf_file_name(folder_name, idf, ngrams, output_folder)
        pdf_file_name_bar = pdf_file_name.replace(".pdf", "_bars.pdf")
        tex_str = tex_str + """
                            \\begin{frame}
                            \\includegraphics[width=\\textwidth]{"""
        tex_str = tex_str + pdf_file_name
        tex_str = tex_str + "}"
        tex_str = tex_str + "\\end{frame}"
        
        bar_tex_str = bar_tex_str + """
                            \\begin{frame}
                            """
        bar_tex_str = bar_tex_str + """
                            \\begin{center}
                            \\includegraphics[width=0.8\\textwidth]{"""
        bar_tex_str = bar_tex_str + pdf_file_name_bar
        bar_tex_str = bar_tex_str + """}
                            \\end{center}
                            \\end{frame}
                            """
                            

    all_file_name = construct_pdf_file_name(ALL_FOLDERS_NAME, idf, ngrams, output_folder)
    pdf_file_name_bar = all_file_name.replace(".pdf", "_bars.pdf")
    texts.append("\n".join(texts_in_all_folders)) # all files in all folders to one text
    names.append(ALL_FOLDERS_NAME)
    tex_str = tex_str + """
                            \\begin{frame}
                            \\includegraphics[width=\\textwidth]{"""
    tex_str = tex_str + all_file_name
    tex_str = tex_str + "}"
    tex_str = tex_str + "\\end{frame}"
    
    bar_tex_str  = bar_tex_str  + """
                            \\begin{frame}
                        """
    bar_tex_str = bar_tex_str + """
                            \\begin{center}
                            \\includegraphics[width=0.8\\textwidth]{"""
    bar_tex_str = bar_tex_str + pdf_file_name_bar
    bar_tex_str = bar_tex_str + """}
                            \\end{center}
                            \\end{frame}
                            """
    
    tex_str = tex_str + "\\end{document}"
    bar_tex_str = bar_tex_str + "\\end{document}"
    tex_output = os.path.join(output_folder, folder_title + get_extra_indicator(idf, ngrams, output_folder) + ".tex")
    bar_tex_output = tex_output.replace(".tex", "_bar.tex")
    with open(tex_output, "w") as write_tex:
        write_tex.write(tex_str)
    with open(bar_tex_output, "w") as write_tex_bar:
        write_tex_bar.write(bar_tex_str)
        
    # Visualise the texts
    
    # To be able to run vectorize_and_generate_plotdata with a different list of terms to emphasise
    # in each run. Here the same list is used in all runs
    terms_to_emphasize_list_duplicates = []
    for i in range(0, len(texts)):
        terms_to_emphasize_list_duplicates.append(terms_to_emphasize_list)
    
    vectorize_and_generate_plotdata(background_corpus, texts, names, stopwords, word2vec_model, ngrams, nr_of_words_to_show, output_folder, x_length_factor, idf, extra_words, y_length_factor, fontsize, same_y_axis, mark_new_words, bar_height_boost, bar_strength_boost, bar_start_boost, min_df, max_df, add_title, terms_to_emphasize_list_duplicates, log_for_bar_height, min_df_in_document, terms_to_emphasize_list_require_exact_match_ngrams, terms_to_emphasize_inverse, scale_max_x_used, min_f_in_current_document, show_number_of_occurrences, plot_mushroom_head, min_tf_in_corpora, lowercase, include_all_terms_to_emphasize, y_extra_space_factor, y_extra_pow_factor, above_below_ratio, use_global_max_score, show_words_outside_ylim, nr_of_vertical_lines)
    # The calculations whether the words collide are made from size plot data that can only be retrieved from matplotlib after the actual plotting is done. Therefore, several plots have to be made, where the exact positions are iteratively adjusted. As a default, one iteration is run.
    plot_from_json(names, idf, ngrams, output_folder, nr_iterations_for_fine_ajusting_the_placement, nr_of_words_in_bar_plot)


def plot_from_json(names, idf, ngrams, output_folder, nr_iterations_for_fine_ajusting_the_placement, nr_of_words_in_bar_plot):
    json_datas = []
    for title in names:
        file_name = os.path.join(output_folder, construct_pdf_file_name(title, idf, ngrams, output_folder))
        with open(file_name + ".json", "rt") as json_file:
            json_datas.append(json_file.read())
    """ # Was used for fine-tuning placement before, but probably not needed
    for i in range(nr_iterations_for_fine_ajusting_the_placement):
        print("Fine adjustment iteration: ", i)
        for title,(i,json_data) in zip(names,enumerate(json_datas)):
            json_datas[i] = plot_round(json_datas[i])
    """
    plot_from_plotdata(names, idf, ngrams, output_folder, json_datas, nr_of_words_in_bar_plot)
