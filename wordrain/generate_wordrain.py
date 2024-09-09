from __future__ import annotations

from . import FONT_NORMAL, FONT_EMPHASIZED, FONT_NEWWORD, DEFAULT_FONTSIZE_FALLOFF_RATE

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
import matplotlib.pyplot as pyplot
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
from typing import NamedTuple, Any
from types import SimpleNamespace
from matplotlib import font_manager as fm
import struct
from io import BytesIO
from collections import Counter
import re
from itertools import chain, islice
from . import pdfgen
from .types import WordScore, WordInfo
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestCentroid
import time
import sys

def flatten_iter(l):
    return chain.from_iterable(l)

NR_OF_WORDS_TO_SHOW = 300
NGRAMS = (1,2)
FONTSIZE = 16
X_STRETCH = 30 # How much to make the x-coordinates wider
DEBUG = False
DEBUG_VECTORS = False
letters = list(string.ascii_lowercase) + ['å', 'ä', 'ö', 'ü', 'é', 'ß']
ALL_FOLDERS_NAME = "all"

#https://stackoverflow.com/questions/24581194/matplotlib-text-bounding-box-dimensions

#matplotlib.rcParams["font.family"] = "monospace"
#plt.rcParams["pdf.use14corefonts"] = True
#matplotlib.rc("text", usetex=True)

pdfBackendEnabled = True

def choose_fonts(fontlist):
    global pdfBackendEnabled
    pdfBackendEnabled = True
    pdfgen.registerFonts(fontlist)

def disablePdfBackend():
    global pdfBackendEnabled
    pdfBackendEnabled = False

class Point(NamedTuple):
    x: float
    y: float
    
    def scalex(self, scale):
        return Point(self.x * scale, self.y)

    def textforward(self, width, params):
        if params.rtl_text:
            return Point(self.x - width, self.y)
        else:
            return Point(self.x + width, self.y)

    def addy(self, height):
        return Point(self.x, self.y + height)

class Rect():
    min: Point
    max: Point

    def __init__(self, p0, p1):
        if p0.x < p1.x:
            x_min = p0.x
            x_max = p1.x
        else:
            x_min = p1.x
            x_max = p0.x
        if p0.y < p1.y:
            y_min = p0.y
            y_max = p1.y
        else:
            y_min = p1.y
            y_max = p0.y
        self.min = Point(x_min, y_min)
        self.max = Point(x_max, y_max)

    def intersects(self, other):
        other_in_me_x = self.min.x <= other.min.x <= self.max.x
        me_in_other_x = other.min.x <= self.min.x <= other.max.x
        other_in_me_y = self.min.y <= other.min.y <= self.max.y
        me_in_other_y = other.min.y <= self.min.y <= other.max.y
        return (other_in_me_x or me_in_other_x) and (other_in_me_y or me_in_other_y)

    def padded(self, top, right, bottom, left):
        return Rect(Point(self.min.x-left, self.min.y-bottom), Point(self.max.x+right, self.max.y+top))

    def __str__(self):
        return "Rect(%f,%f)-(%f,%f)" % (self.min.x, self.min.y, self.max.x, self.max.y)

class WordPlotInfo(NamedTuple):
    word: str
    score: float
    x: float
    fontsize: float
    freq: float
    relfreq: float
    show: str

    @classmethod
    def create(cls, word, score, max_score, show, params):
        return WordPlotInfo(word.word, score, word.x, fontsize_from_score(params, score, max_score), word.freq, word.relfreq,
                                show)

roman_ten_figures = ",X,XX,XXX,XL,L,LX,LXX,LXXX,XC".split(",")
roman_one_figures = ",I,II,III,IV,V,VI,VII,VIII,IX".split(",")

def convert_to_roman_plus_one(n):
    n = n + 1 # start with 1 instead of 0
    
    tens = roman_ten_figures[(n % 100) // 10]
    ones = roman_one_figures[n % 10]
 
    return "%s%s" % (tens, ones)
    
def clean_folder_name(output_folder):
    return output_folder.replace("/", "-").replace("_", "-")
        
        
def latex_clean(text):
    text = text.replace("_", " ")
    text = ''.join(ch for ch in text if ch.isalnum() or ch in [".", ",", " ", "!", "?", "-", "/"]).strip()
    return text
    
def construct_pdf_file_name(input_name, output_folder):
    return input_name.lower().replace(" ", "-").replace(":", "_").replace("å", "ao").replace("ö", "oo").replace("ä", "aa") + "-" + clean_folder_name(output_folder) + ".pdf"
  
def word_in_terms_to_emphasize(word, params):
    emphasize_term = word_in_terms_to_emphasize_before_inverse(word, params)
    if params.terms_to_emphasize_inverse:
        return not emphasize_term
    else:
        return emphasize_term

def word_in_terms_to_emphasize_before_inverse(word, params):
    if params.terms_to_emphasize_list_require_exact_match_ngrams:
        return word in params.terms_to_emphasize or word.replace("_", " ") in params.terms_to_emphasize
    else:
        if word in params.terms_to_emphasize:
            return True
        else:
            emphasized_subwords = [sp in params.terms_to_emphasize for sp in word.split(" ")]
            return any(emphasized_subwords)

def lowercase_terms_to_emphasize(terms_to_emphasize):
    return [term.lower() for term in terms_to_emphasize]

# Very simple compound spitting, not really language independent, but also not very well adapted to any language
def get_compound_vector(word, word2vec_model):
    return_vector = None
    for i in range(4, len(word) - 3):
        first = word[:i].lower()
        second = word[i:].lower()
        if first in word2vec_model and second in word2vec_model:
            second_v = word2vec_model[second]
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


def default_color_map(cmap_value):
    cmap = matplotlib.colormaps['cool']
    color_base = cmap(cmap_value)
    midpoint = 0.5
    color_base_hsv = matplotlib.colors.rgb_to_hsv(color_base[0:3])
    color_base_hsv[1] = 0.6 + abs(midpoint-cmap_value)/3
    color_base_hsv[2] = 1.0 - abs(midpoint-cmap_value)/5
    color_base = list(matplotlib.colors.hsv_to_rgb(color_base_hsv)) + [1.0]
    return color_base

def get_word_to_display(word, title, new_words, params):
    fontstyle = "normal"
    fontweight = "normal"
    word_to_display = word.word.replace(" ", "_")
    if params.transform_displayed_word is not None:
        word_to_display = params.transform_displayed_word(word_to_display)
    word_to_display_wo_markers = word_to_display
    if title != ALL_FOLDERS_NAME:
        if params.mark_new_words and word.word in new_words:
            # First time it appears
            word_to_display = word_to_display + "#"
            fontstyle = "italic" # before used for indicating found in all texts
    selected_lexicon_term = word_in_terms_to_emphasize(word.word, params)
    if selected_lexicon_term:
        fontweight = "bold"
    if params.show_number_of_occurrences:
        if isinstance(params.show_number_of_occurrences, float):
            number_to_show = round(1000*word.relfreq, 1)
        elif isinstance(params.show_number_of_occurrences, int):
            number_to_show = int(word.freq)
        else:
            print("Unknown type of show_number_of_occurrences")
            exit()
        if number_to_show <= params.show_number_of_occurrences:
            word_to_display = word_to_display + "(" + str(number_to_show) + ")"
    word_to_display = get_display(word_to_display)
    word_to_display_wo_markers = get_display(word_to_display_wo_markers)
    return word_to_display, word_to_display_wo_markers, fontstyle, fontweight, selected_lexicon_term

def get_fontsize_falloff_rate(params, title):
    # To be able to set fontsize_falloff_rate individually for each plot
    fontsize_falloff_rate = params.fontsize_falloff_rate
    
    if isinstance(fontsize_falloff_rate, dict):
        fontsize_falloff_rate = {unicodedata.normalize("NFC", key):item for (key, item) in fontsize_falloff_rate.items()}

        if title in fontsize_falloff_rate:
            fontsize_falloff_rate = fontsize_falloff_rate[title]
        else:
            fontsize_falloff_rate = fontsize_falloff_rate.get(DEFAULT_FONTSIZE_FALLOFF_RATE, 1.0)
    return fontsize_falloff_rate

def fontsize_from_score(params, score, max_score):
    fontsize_to_use = params.fontsize*score/max_score

    if fontsize_to_use < 1:
        fontsize_to_use = 1 # if it goes below 1, it is 1 anyway.
    return fontsize_to_use

def set_limits(axes, min_all_x, max_all_x, max_score, params):
    ymin_for_ylim_without_margin = -max_score*(1/(params.bar_ratio)-1)
    ymin_for_ylim = ymin_for_ylim_without_margin*params.margin
    min_all_x *= X_STRETCH
    max_all_x *= X_STRETCH
    graph_width = max_all_x - min_all_x
    xmin_for_xlim = min_all_x - graph_width*params.left_margin
    xmax_for_xlim = max_all_x + graph_width*params.right_margin*params.scale_max_x_used
    axes.set_xlim(xmin=xmin_for_xlim, xmax=xmax_for_xlim)
    axes.set_ylim(ymin=ymin_for_ylim, ymax=max_score*params.margin)
    return ymin_for_ylim_without_margin, ymin_for_ylim, min_all_x, max_all_x, xmin_for_xlim, xmax_for_xlim

def calculate_word_frequencies_sum(freq_dict):
    # Frequencies, but remove n-grams
    all_word_frequencies = [v for (k,v) in freq_dict.items() if " " not in k]
    return sum(all_word_frequencies)

def do_plot(fig, sorted_word_scores_vec, title, min_x, max_x, max_score, new_words, params, actual_plotted_word_size_dict=None):
    emphasis_color = (0.45, 0.45, 0.45)
    emphasis_color_down = (0, 0, 0, 0.2)
    
    print("Do plot for: ", title, "(second run)")

    cmap = default_color_map
    fig.gca().set_axis_off()
    
    # To avoid auto-scaling, set limits for the axes
    
    if params.fontpath is not None:
        fontproperties = fm.FontProperties(fname=params.fontpath)
    else:
        fontproperties = {}

    axes = fig.gca()
    ymin_for_ylim_without_margin, ymin_for_ylim, min_all_x, max_all_x, xmin_for_xlim, xmax_for_xlim = set_limits(axes, min_x, max_x, max_score, params)
    
    for vertical_line_nr in range(0, params.nr_of_vertical_lines):
        vertical_line_position = xmin_for_xlim + (xmax_for_xlim - xmin_for_xlim)/1000 +  vertical_line_nr*(xmax_for_xlim-xmin_for_xlim)/params.nr_of_vertical_lines
        axes.axvline(vertical_line_position, ymin_for_ylim, max_score*params.margin,  linestyle=params.vertical_line_style, zorder = -40000000, linewidth=params.vertical_line_width, color=(0,0,0))
        
    texts = []
    z_order_text = 0
    word_nr = 1
        
    previous_points = []

    for word in sorted_word_scores_vec:
        text_color = (0.0, 0.0, 0.0)
        fontsize_to_use = word.fontsize

        word_to_display, word_to_display_wo_markers, fontstyle, fontweight, selected_lexicon_term = word.show

        y_extra_space_factor_to_use = params.y_extra_space_factor
        if y_extra_space_factor_to_use < 1:
            y_extra_space_factor_to_use = 1
            
        # If the word is underlined, a little extra space is needed
        if selected_lexicon_term:
            y_extra_space_factor_to_use = y_extra_space_factor_to_use*1.2

        len_of_word = len(word_to_display)

        point = Point(word.x, 0)

        point = point.scalex(X_STRETCH)

        changed = True

        assert(word.word in actual_plotted_word_size_dict)
        bbox_x = actual_plotted_word_size_dict[word.word][0]
        margin_x = bbox_x * (params.x_extra_space_factor-1)
        max_point = point.textforward(bbox_x, params)

        assert(word.word in actual_plotted_word_size_dict)
        ystep = actual_plotted_word_size_dict[word.word][1] * y_extra_space_factor_to_use
        while changed:
            point_y_before_loop = point.y
            for prev_word, prev_point in previous_points:

                left_upper_max_point_y = point.y + ystep

                my_bounding_box = Rect(point, Point(max_point.x, left_upper_max_point_y)).padded(0,margin_x,0,margin_x)
                if my_bounding_box.intersects(prev_point):
                    point = Point(point.x, prev_point.max.y)
            if point_y_before_loop == point.y:
                changed = False


        y_bar_start = (max_score - ymin_for_ylim)*0.01*params.bar_start_boost


        cmap_value = (point.x - min_all_x)/(max_all_x - min_all_x)

        if params.color_map is not None:
            color_base = params.color_map(cmap_value)
        else:
            color_base = cmap(cmap_value)
        color_lighter_value = 0.2 #min(0.15*fontsize_to_use, 1)
        color_lighter = (color_base[0], color_base[1], color_base[2], color_lighter_value)
        color_lighter_lighter_value = 0.1 #min(0.1*fontsize_to_use, 1)
        color_lighter_lighter = (color_base[0], color_base[1], color_base[2], color_lighter_lighter_value)

        marker=MarkerStyle('v', joinstyle=JoinStyle.miter)

        # The bar upwards
        y_bar_length = word.score
        linewidth = params.bar_strength_boost*word.score/max_score

        upwardbar_end = y_bar_start + y_bar_length
        color_of_upward_bar = color_base



        # Make lines for terms to emphasize in black
        if selected_lexicon_term:
            color_of_upward_bar = emphasis_color
            upward_bar_zorder = -z_order_text
            mushroom_head_zorder = -z_order_text
            mushroom_head_color = emphasis_color
            mushroom_head_marker = marker
        # Don't mark with grey if there are terms_to_emphasize
        elif params.mark_new_words and not params.terms_to_emphasize and word.word in new_words:
            color_of_upward_bar = emphasis_color
            upward_bar_zorder = -z_order_text
            mushroom_head_zorder = -z_order_text
            mushroom_head_color = emphasis_color
            mushroom_head_marker = "*"
        else:
            upward_bar_zorder = -2000-z_order_text
            mushroom_head_zorder = -200000
            mushroom_head_color = color_of_upward_bar
            mushroom_head_marker = "o"
        if params.draw_vertical_bars:
            axes.plot([point.x, point.x], [y_bar_start, upwardbar_end], color=color_of_upward_bar, zorder = upward_bar_zorder, linewidth=linewidth)
            if params.plot_mushroom_head:
                axes.scatter(point.x, upwardbar_end, zorder = mushroom_head_zorder, color=mushroom_head_color, marker = mushroom_head_marker, s=linewidth)



        # The texts have been changed to be printed downwards instead
        negative_text_y = -point.y

        horizontalalignment = get_horizontalalignment(params)
        t = None
        if params.show_words_outside_ylim or negative_text_y >= ymin_for_ylim_without_margin:
            if params.background_box:
                background_color = color_lighter_lighter
            else:
                background_color = None
            t = axes.text(point.x, negative_text_y, word_to_display, zorder = z_order_text, color = text_color, fontsize=fontsize_to_use, fontstyle=fontstyle, fontweight=fontweight, verticalalignment="top", fontproperties=fontproperties, horizontalalignment=horizontalalignment, background=background_color)
            if selected_lexicon_term:
                underline_str = len(word.word)*"_"

                if params.transform_displayed_word:
                    underline_str = len(params.transform_displayed_word(word.word))*"_"
                try:
                    axes.underline(point.x, negative_text_y, word_to_display_wo_markers, zorder = z_order_text, color = text_color, fontsize=fontsize_to_use, fontstyle=fontstyle, fontweight=fontweight, verticalalignment="top", fontproperties=fontproperties, horizontalalignment=horizontalalignment)
                except AttributeError:
                    t_underline = axes.text(point.x, negative_text_y, underline_str, zorder = -20000, color = (0.0, 0.0, 0.0), fontsize=fontsize_to_use, verticalalignment="top", horizontalalignment=horizontalalignment)
        else:
            if word.word in new_words and params.mark_new_words:
                # Add new words at the border, but don't make them bold (just for estetical reasons)
                axes.text(point.x, ymin_for_ylim_without_margin*1.01, word_to_display, zorder = z_order_text, color = emphasis_color, fontsize=1, fontstyle=fontstyle, fontweight="normal", verticalalignment="top", rotation=270, fontproperties=fontproperties, horizontalalignment=horizontalalignment)
            if params.draw_vertical_bars:
                axes.scatter(point.x, ymin_for_ylim_without_margin, zorder = -200000, color=color_lighter_lighter, marker = ".", s=fontsize_to_use*0.01)
            negative_text_y = ymin_for_ylim_without_margin # to make the bar downward stop

        if t is not None:
            x_length_from_bbox, y_length_from_bbox = calculate_text_size(t, fig, axes)

            max_point = (point
                .textforward(x_length_from_bbox, params)
                .addy(y_length_from_bbox*y_extra_space_factor_to_use)
            )
            margin_x = x_length_from_bbox * (params.x_extra_space_factor-1)
            previous_points.append((word_to_display, Rect(point, max_point).padded(0,margin_x,0,margin_x)))



        # The bar downwards
        if params.fixed_linewidth_down is None: # Then make it in relation to the bar strength
            linewidth_down=params.bar_strength_boost*word.score/max_score
        else:
            linewidth_down = params.fixed_linewidth_down

        linestyle_down = "solid"
        if not params.draw_vertical_bars:
            pass
        elif selected_lexicon_term:
            axes.plot([point.x, point.x], [y_bar_start, negative_text_y], color=emphasis_color_down, linewidth=linewidth_down, zorder = -10000000, linestyle=linestyle_down)
            axes.scatter(point.x, negative_text_y, zorder = -100000, color=emphasis_color_down, marker = marker, s=fontsize_to_use*0.01) #
        elif params.mark_new_words and not params.terms_to_emphasize and word.word in new_words:
            axes.plot([point.x, point.x], [y_bar_start, negative_text_y], color=emphasis_color_down, linewidth=linewidth_down, zorder = -30000000, linestyle=linestyle_down)
            axes.scatter(point.x, negative_text_y, zorder = -300000, color=emphasis_color_down, marker = marker, s=fontsize_to_use*0.01) #
        else:
            axes.plot([point.x, point.x], [y_bar_start, negative_text_y], color=color_lighter, linewidth=linewidth_down, zorder = -20000000, linestyle=linestyle_down)
            axes.scatter(point.x, negative_text_y, zorder = -200000, color=color_lighter_lighter, marker = "o", s=fontsize_to_use*0.01) #


        z_order_text =  z_order_text - 1
        word_nr = word_nr + 1
        
        if title == ALL_FOLDERS_NAME and word_nr > 3000:
            break  # To avoid that this one getting really large, when nr of top words is None

    if params.plot_vertical_line_label:
        if params.compact:
            line_label_y = -max(bbox.max.y for _, bbox in previous_points)+ymin_for_ylim/100
        else:
            line_label_y = ymin_for_ylim
        for vertical_line_nr in range(0, params.nr_of_vertical_lines):
            vertical_line_position = xmin_for_xlim + (xmax_for_xlim - xmin_for_xlim)/1000 +  vertical_line_nr*(xmax_for_xlim-xmin_for_xlim)/params.nr_of_vertical_lines
            axes.text(vertical_line_position + (xmax_for_xlim - xmin_for_xlim)/1000, line_label_y, convert_to_roman_plus_one(vertical_line_nr), zorder = -30000000, fontsize=2, fontproperties=fontproperties)

    title_to_use = "\n" + title
    title_to_use = latex_clean(title_to_use)
    title_fontsize = 10/math.log(len(title_to_use) + 3)*1.6
    if params.add_title:
        axes.set_title(title_to_use, fontsize=title_fontsize)
    
def get_horizontalalignment(params):
    if params.rtl_text:
        return "right"
    else:
        return "left"

def calculate_text_size(t, fig, axes):
    bbox = t.get_window_extent(renderer = fig._get_renderer())
    transf = axes.transData.inverted()
    bb_datacoords = bbox.transformed(transf)
    x_length_from_bbox = bb_datacoords.x1 - bb_datacoords.x0
    y_length_from_bbox = bb_datacoords.y1 - bb_datacoords.y0
    return x_length_from_bbox, y_length_from_bbox


def do_plot_prepare(fig, sorted_word_scores_vec, title, max_score, new_words, min_x, max_x, params):
    if params.fontpath is not None:
        fontproperties = fm.FontProperties(fname=params.fontpath)
    else:
        fontproperties = {}

    axes = fig.gca()
    set_limits(axes, min_x, max_x, max_score, params)

    word_nr = 1

    actual_plotted_word_size_dict_to_create = {}

    actual_text_boundary_xs = []

    for word in sorted_word_scores_vec:
        start_point = Point(word.x, 0).scalex(X_STRETCH)

        text_color = (0.0, 0.0, 0.0)
        word_to_display, _, fontstyle, fontweight, _ = word.show
        t = axes.text(0, 0, word_to_display, zorder = 0, color = text_color, fontsize=word.fontsize, fontstyle=fontstyle, fontweight=fontweight, verticalalignment="top", fontproperties=fontproperties, horizontalalignment=get_horizontalalignment(params))
        if t is not None:
            x_length_from_bbox, y_length_from_bbox = calculate_text_size(t, fig, axes)

            actual_plotted_word_size_dict_to_create[word.word] = (x_length_from_bbox, y_length_from_bbox)
            end_point = start_point.textforward(x_length_from_bbox, params)
            actual_text_boundary_xs.append(start_point.x)
            actual_text_boundary_xs.append(end_point.x)

        word_nr = word_nr + 1

        if title == ALL_FOLDERS_NAME and word_nr > 3000:
            break  # To avoid that this one getting really large, when nr of top words is None

    return actual_plotted_word_size_dict_to_create, min(actual_text_boundary_xs), max(actual_text_boundary_xs)

def get_word_x_from_hash(word, min_x, max_x):
    sha1hash = hashlib.sha1(word.encode('utf-8')).digest()
    (uint64_hash,) = struct.unpack("!Q", sha1hash[:8])
    normalized_hash = math.ldexp(uint64_hash, -64)
    x_length = max_x - min_x
    x_point = normalized_hash * x_length + min_x
    print(sha1hash, uint64_hash, normalized_hash, x_point)
    return x_point

def write_scores(output_folder, sorted_word_scores_matrix, names, new_words_lists, extra_saves):
    # Create folder for storing tf-idf or tf statistics
    texts_folder = "texts-" + clean_folder_name(output_folder)
    texts_output = os.path.join(output_folder, texts_folder)
    if not os.path.exists(texts_output):
        os.mkdir(texts_output)

    # Write scores to file
    for sws, title in zip(sorted_word_scores_matrix, names):

        tf_name = os.path.join(texts_output, title + "-" + clean_folder_name(output_folder) + ".txt")
        with open(tf_name, "w") as tf_write:
            for sw in sws:
                w_freq_proportional = round(1000*sw.relfreq, 1)

                tf_write.write("\t".join([sw.word, str(sw.score), "(%.1f)" % sw.freq, "(%.1f)" % w_freq_proportional, str([sw.x,0])]) + "\n")

    for new_words_list, title in zip(new_words_lists, names):
        if title != ALL_FOLDERS_NAME:
            new_words_output = texts_output + "-new-words"
            if not os.path.exists(new_words_output):
                os.mkdir(new_words_output)
            new_words_file_name = os.path.join(new_words_output, "new_words_" + title + "-" + clean_folder_name(output_folder) + ".txt")
            with open(new_words_file_name, "w") as nwfn:
                for new_word_el in sorted(new_words_list):
                    nwfn.write(new_word_el + "\n")

    for extra_save, filename in extra_saves:
        with open(os.path.join(texts_output, filename % (title,)), "wt") as f:
            for entry in extra_save:
                print(entry, file=f)

def include_emphasized(word, params):
    return params.include_all_terms_to_emphasize and word_in_terms_to_emphasize(word, params)
    

subword_re = re.compile("[- ]")

def subword_match(word, wordlist):
    subwords = subword_re.split(word)
    matched_subwords = [subword in wordlist for subword in subwords]
    return any(matched_subwords)

def extra_kwargs_for_TfidfVectorizer(params):
    extra_kwargs = {}
    if params.token_pattern is not None:
        extra_kwargs["token_pattern"] = params.token_pattern
    return extra_kwargs

def vectorize_max_df(texts, stopwords, params):
    max_df_vectorizer = TfidfVectorizer(stop_words=stopwords, min_df=params.min_df, max_df=params.max_df, smooth_idf=False, sublinear_tf=False, ngram_range = params.ngrams, use_idf=params.idf, norm=None, lowercase=params.lowercase, **extra_kwargs_for_TfidfVectorizer(params))
    print("nr of texts", len(texts))
    start_time = time.time()
    max_df_X = max_df_vectorizer.fit_transform(texts)
    print("fit_transform max_df_set", time.time()-start_time)
    max_df_set = set(max_df_vectorizer.vocabulary_)

    return max_df_set

# A local vectorizer for just each text, to count frequencies in each text and save in freq_dict_list
def vectorize_one_text(text, stopwords, params):
    freqs_for_entire_corpora = Counter()
    freq_vectorizer = TfidfVectorizer(sublinear_tf=False, ngram_range = params.ngrams, use_idf=False, norm=None, stop_words=stopwords, lowercase=params.lowercase, **extra_kwargs_for_TfidfVectorizer(params))
    start_time = time.time()
    freq_X = freq_vectorizer.fit_transform([text])
    print("fit_transform onetext", time.time()-start_time)
    freqency_inversed = freq_vectorizer.inverse_transform(freq_X)
    freq_list = []
    freq_dict = {}
    for words, freqs in zip(freqency_inversed, freq_X):
        score_vec = freqs.toarray()[0]
        for word in words:
            freq_for_word = score_vec[freq_vectorizer.vocabulary_[word]]
            freq_dict[word] = freq_for_word

            # To count the total frequency for a word, in all texts

            freqs_for_entire_corpora[word] += freq_for_word
    return freq_dict, freqs_for_entire_corpora

def vectorize_corpus(texts, names, word2vec_model, output_folder, params):
    stopwords = list(params.stopwords)

    # For tf-idf
    ############
        
    freq_dict_list = [] # A list of word freqencies-dictionaries for each document
    freqs_for_entire_corpora = Counter()

    vectorize_one_text_results = map(lambda text: vectorize_one_text(text, stopwords, params), texts)
    for freq_dict, count in vectorize_one_text_results:
        freqs_for_entire_corpora.update(count)
        freq_dict_list.append(freq_dict)

    # Create a vocabulary where all words with a frequencey < min_tf_in_corpora are excluded
    vocabulary_to_use = set([word for (word, freq) in freqs_for_entire_corpora.items() if freq >= params.min_tf_in_corpora])

    if params.max_df != 1.0 or params.min_df > 1:
        print("before, max_df, min_df",  len(vocabulary_to_use))
        max_df_set = vectorize_max_df(texts, stopwords, params)
        vocabulary_to_use &= max_df_set
        print("after, max_df, min_df", len(vocabulary_to_use))
        
    print("before",  len(vocabulary_to_use))
    if params.include_all_terms_to_emphasize:
        vocabulary_to_use |= set(params.terms_to_emphasize)
    print("after", len(vocabulary_to_use))
    
    # Handle if the vocabulary is restricted
    if params.restrict_vocabulary_to_this_list and params.restrict_vocabulary_requires_exact_match:
         vocabulary_to_use &= set(params.restrict_vocabulary_to_this_list)
    if params.restrict_vocabulary_to_this_list and not params.restrict_vocabulary_requires_exact_match:
        vocabulary_to_use = [voc for voc in vocabulary_to_use if subword_match(voc, params.restrict_vocabulary_to_this_list)]
    
    # The main vectorizer, with the statistics to use for plotting
    main_vectorizer = TfidfVectorizer(stop_words=stopwords, min_df=params.min_df, max_df=params.max_df, smooth_idf=False, sublinear_tf=False, ngram_range = params.ngrams, use_idf=params.idf, norm=None, vocabulary = vocabulary_to_use, lowercase=params.lowercase, **extra_kwargs_for_TfidfVectorizer(params))

    print("len(texts), all texts inclusive", len(texts))
    print("len(texts + background_corpus)", len(texts + params.background_corpus))

    start_time = time.time()
    main_vectorizer.fit_transform(texts + params.background_corpus)
    print("fit_transform final", time.time()-start_time)
    start_time = time.time()
    X = main_vectorizer.transform(texts)
    print("fit_transform final transform", time.time()-start_time)
    inversed = main_vectorizer.inverse_transform(X)
    vocabulary = main_vectorizer.vocabulary_
    
    word_scores_per_graph_before_cutoff = []
    for el, w, freq_dict in zip(inversed, X, freq_dict_list):
        sum_all_word_frequencies = calculate_word_frequencies_sum(freq_dict)
        score_vec = w.toarray()[0]
        word_score_vec = [
            WordScore(score_vec[vocabulary[word]], word, include_emphasized(word, params), freq_dict[word], freq_dict[word]/sum_all_word_frequencies)
            for word in el
        ]
        word_scores_per_graph_before_cutoff.append(word_score_vec)
         
    # Create space (and score statistics):
    
        
    # Only plot top nr_of_words_to_show words
    # And also, filter out too infrequent words (from the parameter min_f_in_current_document)
    # Here filtering on text-level is performed (in contrast to corpora level)
    # Also apply the nr_of_words_to_show cut-off
    
    # Make configurable transformation, if there is one
    if params.transform_word_scores:
        word_scores_per_graph_before_cutoff = params.transform_word_scores(word_scores_per_graph_before_cutoff)
        
    assert not any(sw.score < 0 for sw in flatten_iter(word_scores_per_graph_before_cutoff)), "Scores need to be positive"

    sorted_word_scores_per_graph = []
    for word_scores in word_scores_per_graph_before_cutoff:
        word_scores = sorted(word_scores, key=lambda sw: (sw.force_inclusion, sw.score, sw.word), reverse=True)
        sorted_word_scores_per_graph.append(sorted(
            islice(
                filter(
                    lambda sw: sw.force_inclusion or sw.freq >= params.min_f_in_current_document,
                    word_scores),
                params.nr_of_words_to_show),
            reverse=True))

    word_to_use_set = set()
    new_words_lists = []
    assert len(names) == 1 or names[-1] == ALL_FOLDERS_NAME
    for word_scores in sorted_word_scores_per_graph:
        words = {sw.word for sw in word_scores}
        # Calculate difference between this corpus and previous (cut-off-filtered) corpora
        new_words_lists.append(words - word_to_use_set)
        word_to_use_set |= words

    all_vectors_list = []
    found_words = []
    not_found_words = []

    
    # It is possible to add extra words, not found in the corpus, to be
    # part of the projection, as might be needed for other visualisations
    words_to_generate_space_for = word_to_use_set
    if params.extra_words:
        words_to_generate_space_for = words_to_generate_space_for.union( set(params.extra_words))
        
    words_to_generate_space_for = sorted(list(set(words_to_generate_space_for)))
    print("Nr of words to generate space for:", len(words_to_generate_space_for))
    
    for word in words_to_generate_space_for:
        # Add vectors
        norm_vector = get_vector_for_word(word, word2vec_model, params.ngrams)
        if norm_vector == None:
            not_found_words.append(word)
        else:
            found_words.append(word)
            all_vectors_list.append(norm_vector)
                    
    all_vectors_np = np.array(all_vectors_list)
    print("len(all_vectors_list)", len(all_vectors_list))
    
    
    extra_saves = []

    if DEBUG_VECTORS:
        extra_saves.append((all_vectors_list, "word-vectors-%s.txt"))
        extra_saves.append((found_words, "words-%s.txt"))

    if params.use_saved_vectors:
        DX = np.load(params.use_saved_vectors)
    else:
        n_components = min(50, len(all_vectors_list))
        if params.nr_of_clusters is None:
            pca_model = PCA(n_components=n_components, random_state=0)
            tsne_model = TSNE(n_components=1, random_state=0, perplexity=min(30, n_components-1))
            DX_pca = pca_model.fit_transform(all_vectors_np)
            if DEBUG_VECTORS:
                extra_saves.append((DX_pca, "word-vectors-pca-%s.txt"))
            DX = tsne_model.fit_transform(DX_pca)
            if DEBUG_VECTORS:
                extra_saves.append((DX, "word-vectors-tsne-%s.txt"))
        else:
            agg_clustering = AgglomerativeClustering(n_clusters=params.nr_of_clusters)
            agg_DX_clustering = agg_clustering.fit_predict(all_vectors_np)
            clf = NearestCentroid()
            clf.fit(all_vectors_np, agg_DX_clustering)
            centroids = [clf.centroids_[i] for i in agg_DX_clustering]
            distances = [(orig - centroid)/2 for (orig, centroid) in zip(all_vectors_np, centroids)]
            new_vectors_moved_towards_centroid = np.asarray([orig - distance for (orig, distance) in zip(all_vectors_np, distances)])
            tsne_model = TSNE(n_components=1, random_state=0, perplexity=min(50, n_components-1))
            DX = tsne_model.fit_transform(new_vectors_moved_towards_centroid)
    if params.save_vectors:
        save_vectors_in = os.path.join(output_folder, "saved_tsne_projection.npy")
        print("save tsne projection in", save_vectors_in)
        np.save(save_vectors_in, DX)
        
    word_vec_dict = {}
    for point, found_word in zip(DX, found_words):
        word_vec_dict[found_word] = float(point[0])
    max_x = max(word_vec_dict.values())
    min_x = min(word_vec_dict.values())

    # Add a point on the x axis for words not found in the space, based on the characters in the word
    for word in not_found_words:
        print("Use random point for ", word)
        x_point = get_word_x_from_hash(word, min_x, max_x)
        word_vec_dict[word] = x_point

    word_info_matrix = [
        [
            WordInfo(ws.score, ws.word, word_vec_dict[ws.word], ws.freq, ws.relfreq)
            for ws in word_scores
        ]
        for word_scores in sorted_word_scores_per_graph
    ]

    if output_folder is not None:
        write_scores(output_folder, word_info_matrix, names, new_words_lists, extra_saves)

    # Apart from writing new words to file (as is done above), the first plot should mark no words as new
    new_words_lists[0] = []

    return word_info_matrix, new_words_lists, min_x, max_x

def json_serialize_set(o):
    if isinstance(o, set):
        return sorted(o)
    return o

def generate_plotdata(output_folder, word_info_matrix, names, new_words_lists, min_x, max_x, params):
    max_scores = []
    min_scores = []
    assert isinstance(word_info_matrix[0][0], WordInfo), "word_info_matrix has to be Iterable[Iterable[WordInfo]]"
    for word_scores, title in zip(word_info_matrix, names):
        fontsize_falloff_rate = get_fontsize_falloff_rate(params, title)

        max_score = max([sw.score for sw in word_scores])
        max_score = apply_fontsize_falloff_rate(max_score, fontsize_falloff_rate)
        max_scores.append(max_score)

        min_score = min([sw.score for sw in word_scores])
        min_score = apply_fontsize_falloff_rate(min_score, fontsize_falloff_rate)
        min_scores.append(min_score)


    actual_plotted_word_size_dict_list = []

    if params.use_global_max_score and len(word_info_matrix) > 1:
        max_score_except_all = max(max_scores[:-1])
        for i in range(len(max_scores[:-1])):
            max_scores[i] = max_score_except_all

    word_plot_info_matrix = rewrite_scores(word_info_matrix, max_scores, params, names, new_words_lists)
    
    # Make one first run, just to get the global max and min point (but don't save the result)
    actual_text_boundary_xs = []
    for sorted_word_scores, title, min_score, new_words, plot_nr in zip(word_plot_info_matrix, names, min_scores, new_words_lists, range(0, len(names))):
    
        max_score_to_use_in_round = max_scores[plot_nr]
        #matplotlib.rcParams['figure.figsize'] = [6.4,4.8*2]
        #main_fig = plt.figure(figsize=(6.4,4.8*2))
        if pdfBackendEnabled:
            main_fig = pdfgen.MatplotlibShimPdf(None, pagesize = params.pagesize)
        else:
            main_fig = pyplot.figure()
        main_fig.gca().set_axis_off()

        actual_plotted_word_size_dict, actual_min_x, actual_max_x = \
            do_plot_prepare(main_fig, sorted_word_scores, title, max_score_to_use_in_round, new_words, min_x, max_x, params)
         
        actual_plotted_word_size_dict_list.append(actual_plotted_word_size_dict)

        if params.unified_graph and title == ALL_FOLDERS_NAME:
            pass
        else:
            actual_text_boundary_xs.append(actual_min_x)
            actual_text_boundary_xs.append(actual_max_x)
        
        main_fig.clf()
        if not pdfBackendEnabled:
            pyplot.close('all')

    return word_plot_info_matrix, actual_plotted_word_size_dict_list, max_scores, min(actual_text_boundary_xs), max(actual_text_boundary_xs)


def apply_fontsize_falloff_rate(score, fontsize_falloff_rate):
    if fontsize_falloff_rate:
        return math.pow(score, fontsize_falloff_rate)
    return score

def rewrite_scores(word_info_matrix, max_scores, params, names, new_words_lists):
    fontsize_falloff_rates = [get_fontsize_falloff_rate(params, title) for title in names]
    return [[WordPlotInfo.create(sws, apply_fontsize_falloff_rate(sws.score, fontsize_falloff_rate), max_score, get_word_to_display(sws, title, new_words, params), params) for sws in sorted_word_scores] for sorted_word_scores, fontsize_falloff_rate, max_score, title, new_words in zip(word_info_matrix, fontsize_falloff_rates, max_scores, names, new_words_lists)]

def save_plotdata(output_folder, sorted_word_scores, title, min_x, max_x, max_score_to_use_in_round, new_words, actual_plotted_word_size_dict=None):
    if output_folder is not None:
        file_name = os.path.join(output_folder, construct_pdf_file_name(title, output_folder))
        with open(file_name + ".json", "wt") as json_file:
            json.dump({
                "sorted_word_scores":sorted_word_scores,
                "title":title,
                "min_x": min_x,
                "max_x": max_x,
                "max_score_to_use_in_round": max_score_to_use_in_round,
                "new_words": new_words,
                "actual_plotted_word_size_dict":actual_plotted_word_size_dict
            }, json_file, default=json_serialize_set)

def plot_from_plotdata(output_folder, word_plot_info_matrix, names, new_words_lists, actual_plotted_word_size_dict_list, max_scores, min_x, max_x, actual_min_x, actual_max_x, params):
    # Make one another run and save the results in images

    images = []
    cropboxes = []
    if params.unified_graph:
        assert pdfBackendEnabled
        image = BytesIO()
        if len(names) > 1:
            ngraphs = len(names) - 1
        else:
            ngraphs = 1
        main_fig = pdfgen.MatplotlibShimPdf(image, pagesize = params.pagesize, subgraphs=ngraphs, compact=params.compact)

    for sorted_word_scores, title, new_words, plot_nr, actual_plotted_word_size_dict in zip(word_plot_info_matrix, names,  new_words_lists, range(0, len(names)), actual_plotted_word_size_dict_list):

        max_score_to_use_in_round = max_scores[plot_nr]

        if params.unified_graph:
            if title == ALL_FOLDERS_NAME:
                continue
            main_fig.set_subgraph(plot_nr)
        else:
            image = BytesIO()
            if pdfBackendEnabled:
                main_fig = pdfgen.MatplotlibShimPdf(image, pagesize = params.pagesize, compact=params.compact)
            else:
                main_fig = pyplot.figure()
        if not pdfBackendEnabled:
            pyplot.axis('off')

        save_plotdata(output_folder, sorted_word_scores, title, min_x, max_x, max_score_to_use_in_round, new_words,
                actual_plotted_word_size_dict=actual_plotted_word_size_dict)
        do_plot(main_fig, sorted_word_scores, title, min_x, max_x, max_score_to_use_in_round, new_words, params,
                actual_plotted_word_size_dict=actual_plotted_word_size_dict)
        if pdfBackendEnabled:
            main_fig.register_point_x(actual_min_x)
            main_fig.register_point_x(actual_max_x)

        if not params.unified_graph:
            main_fig.savefig(image,  format="pdf", pad_inches=0)
            image.seek(0)
            images.append(image)
            if pdfBackendEnabled:
                cropboxes.append(main_fig.bbox.rectbox())
            else:
                cropboxes.append(None)
        print("Generated image")
        if not pdfBackendEnabled:
            pyplot.close('all')
    if params.unified_graph:
        main_fig.savefig(image)
        image.seek(0)
        images.append(image)
        cropboxes.append(main_fig.bbox.rectbox())
    return images, cropboxes

def tex_header(folder_title, output_folder):
    tex_str = """
            \\documentclass{beamer}
            \\usepackage[utf8]{inputenc}
            \\beamertemplatenavigationsymbolsempty
            \\usetheme{default}
            \\setbeamertemplate{headline}{}
            """
    tex_str = tex_str + \
            """
            \\title{
            """
    tex_str = tex_str + latex_clean(folder_title + " " + clean_folder_name(output_folder))
    tex_str = tex_str + \
            """
            }
            \\author{}

            \\usepackage{graphicx}
            \\begin{document}

            \\maketitle
                    
            """
    return tex_str

def tex_file_entry(pdf_file_name):
    tex_str = """
                            \\begin{frame}
                            \\vspace*{-0.7cm}\hspace*{-2.1cm}\\includegraphics[width=15.1cm]{"""
    tex_str = tex_str + pdf_file_name
    tex_str = tex_str + "}"
    tex_str = tex_str + "\\end{frame}"
    return tex_str

def generate_tex_file(names, corpus_folder, output_folder):
    pdf_file_names = [construct_pdf_file_name(name, output_folder) for name in names]
    folder_title = os.path.basename(corpus_folder.rstrip("/"))

    tex_str = tex_header(folder_title, output_folder)
    for pdf_file_name in pdf_file_names:
        tex_str += tex_file_entry(pdf_file_name)

    tex_str = tex_str + "\\end{document}"
    tex_output = os.path.join(output_folder, folder_title + "-" + clean_folder_name(output_folder) + ".tex")
    with open(tex_output, "w") as write_tex:
        write_tex.write(tex_str)

def read_corpus_folder(corpus_folder, pre_process_method):
    names = []
    texts = []
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
                    text_in_file = pre_process_method(text_in_file)
                texts_in_folder.append(text_in_file)
        all_text_in_folder = "\n".join(texts_in_folder)
        
        texts.append(all_text_in_folder) # all files in folder to one text
        names.append(folder_name)
    return texts, names


def word_info_restore(word_info_matrix):
    return [
        [
            WordInfo(*word_info)
            for word_info in word_info_list
        ]
        for word_info_list in word_info_matrix
    ]


####
# The main function for generating the word rain
####
#
# scale_max_x_used: the extra space to the right, to make sure all words are in the picture
# min_tf_in_corpora is a global cut-off.
# it removes words that occur less than min_tf_in_corpora in the entire corpus to be plotted anywhere.
# Can for instance be matched with the same cut-off when creating a word2vec-model, to make sure that all words have a matching vector
# min_f_in_current_document is a local cut-off. It excludes words to be plotted that only occur rarely in the document that is plotted. The default is 1.
# fontsize The font size will never be smaller than 1. Therefore, if you use to small font size for the largest words, the word rain will get boring, with most words in font size 1.
# include_all_terms_to_emphasize is default False. When true, all words in all_terms_to_emphasize are plotted (if included in the corpus), regardless of their min-tf and max-tf etc. (but the cut-off of top n, still holds though. So if their prominence is lower than other words, they willl still not be included)
# bar_ratio: Decides how large part of the image will be above 0 (as bars). Default is 1/3, i.e. one third of the page for the bars and two thirds for the words.
# show_number_of_occurrences: A number which dictates the maximum number of occurrences for a word if the number of occurrences is to be shown in paranthesis to the right of the word. If the number is given as an integer, the raw word occurrences is counted. If it is given as a float, the number of occurrences divided by the total number of words in the document is used. The default is False (= don't show any numbers).
# nr_of_words_to_show: The top nr_of_words_to_show. If None, it is set to math.inf. Don't do that if you don't have a very small data set though, because it will be take a very, very long time.
# nr_of_vertical_lines: How many vertical lines to print. Default is 0
# use_global_max_score: If the max score (and also how large the words become on the plot), should be determined locally or globally). Default is False
# fontsize_falloff_rate: The rate by which the font size decreases when the score descreases. A higher value exaggerates the differences between low and high scores, i.e., the font sizes decrease more quickly, which has the effect of make the graphs take up less vertical space. Therefore, if the graph does not fit within the  standard space for a wordrain graph, you might want to increase the fontsize_falloff_rate used. If the words shown are too compressed, you instead might want to decrease the fontsize_falloff_rate.
# extra_words: A list of words to add to the projection. If the same projection is to be used for another visualisation, you might need to add words that are to be included in this future visualisation to the projection
# save_vectors: Save the vectors corresponding to the words visualised, so that they can be reused by another visualisation.
# use_saved_vectors: Load previously saved TSNE projection. The point with this (and "save_vectors" is to be able to have a similar projection for two different visualisations")
# y_extra_space_factor: also, the y-space that a word is allowed to take can be factored by y_extra_space_factor. This is 1 by default.
# fixed_linewidth_down: change the width of the line downwards that connects to the word. Default is 0.1, but e.g. for printing it might need to be thicker. If set to None, it will adapt to the thickness of the bar which it is connected to.



import dataclasses
from collections.abc import Iterable
from collections.abc import Callable

@dataclasses.dataclass(frozen=True)
class Params:
    ngrams: bool = NGRAMS
    nr_of_words_to_show : int | None =  NR_OF_WORDS_TO_SHOW
    stopwords: Iterable[str] = dataclasses.field(default_factory = list)
    background_corpus: list[str] = dataclasses.field(default_factory = list)
    pre_process_method : Callable[[str], str] | None =  None
    idf : bool =  True
    extra_words : Iterable[str] =  dataclasses.field(default_factory = list)
    fontsize: float = FONTSIZE
    mark_new_words: bool = False
    min_df: int|float = 1
    max_df: int|float = 1.0
    add_title: bool = True
    bar_strength_boost: float = 1
    bar_start_boost : float =  1
    terms_to_emphasize_list : Iterable[str] =  dataclasses.field(default_factory = list)
    terms_to_emphasize : Collection[str] = dataclasses.field(init = False)
    fontsize_falloff_rate: float = False
    terms_to_emphasize_list_require_exact_match_ngrams: bool = False
    terms_to_emphasize_inverse: bool = False
    scale_max_x_used: float = 1.3
    min_f_in_current_document: float = 1
    monospace: bool = True
    show_number_of_occurrences: bool = False
    plot_mushroom_head: bool = True
    min_tf_in_corpora: int = 1
    lowercase: bool = True
    include_all_terms_to_emphasize: bool = False
    y_extra_space_factor: float = 1.0
    bar_ratio: float = 1/3
    use_global_max_score: bool = False
    show_words_outside_ylim: bool = False
    nr_of_vertical_lines : int =  0
    save_vectors: bool = False
    use_saved_vectors: str | None = None
    restrict_vocabulary_to_this_list: Iterable[str] = None
    fontpath: str = None
    fixed_linewidth_down: float | None = 0.1
    restrict_vocabulary_requires_exact_match: bool = False
    transform_displayed_word: Callable[[str], str] | None = None
    color_map: Callable[[float], list[float]] = None
    rtl_text: bool = False
    x_extra_space_factor: float = 1.0
    plot_vertical_line_label: bool = True
    draw_vertical_bars: bool = True
    transform_word_scores: list[list[WordScore]] = None
    unified_graph: bool = False
    compact: bool = False
    nr_of_clusters: int | None = None
    background_box: bool = False
    pagesize: (float, float) = (460,345)
    token_pattern: str | None = None
    left_margin : float =  0.001
    right_margin : float =  0.3
    margin : float =  1.1
    vertical_line_width : float =  0.1
    vertical_line_style : str =  "dotted"

    def __post_init__(self):
        if self.lowercase and self.terms_to_emphasize_list:
            object.__setattr__(self, "terms_to_emphasize", lowercase_terms_to_emphasize(self.terms_to_emphasize_list))
        else:
            object.__setattr__(self, "terms_to_emphasize", list(self.terms_to_emphasize_list))

    def updated(self, **kwargs):
        return dataclasses.replace(self, **kwargs)

def generate_clouds_vectorize(corpus_folder, word2vec_model, output_folder, params, corpus_texts=None):
    if corpus_texts is None:
        # Read texts that are to be visualised

        texts, names = read_corpus_folder(corpus_folder, params.pre_process_method)
    else:
        texts = [text for name, text in corpus_texts]
        names = [name for name, text in corpus_texts]

    
    
    if isinstance(params.fontsize_falloff_rate, dict):
        for key in params.fontsize_falloff_rate.keys():
            if (unicodedata.normalize("NFC",key) not in names and key != DEFAULT_FONTSIZE_FALLOFF_RATE):
                print(f"key in fontsize_falloff_rate does not correspond to folder among text files to visualise: {key}")
                print(f"Allowed keys are: {names}")
                print("As well as the variable: 'wordrain.DEFAULT_FONTSIZE_FALLOFF_RATE'")
                sys.exit(1)
            
    if len(texts) > 1:
        names.append(ALL_FOLDERS_NAME)
        texts.append("\n".join(texts)) # all files in all folders to one text

        # To account for the 'all' batch, increase min_df and max_df by one
        if isinstance(params.min_df, int):
            params = params.updated(min_df = params.min_df + 1)
        if isinstance(params.max_df, int):
            params = params.updated(max_df = params.max_df + 1)

    if output_folder is not None:
        generate_tex_file(names, corpus_folder, output_folder)

    # Visualise the texts

    if params.nr_of_words_to_show is None:
        params = params.updated(nr_of_words_to_show = math.inf)
        print("No cut-off is applied for nr_of_words_to_show")


    word_info_matrix, new_words_lists, min_x, max_x = vectorize_corpus(texts, names, word2vec_model, output_folder, params)
    return word_info_matrix, new_words_lists, min_x, max_x, names

def generate_plot(output_folder, word_info_matrix, new_words_lists, min_x, max_x, names, params):
    if params.monospace:
        matplotlib.rcParams["font.family"] = "monospace"
    if pdfBackendEnabled:
        pdfgen.registerDefaultFonts({
            FONT_NORMAL: 'DejaVuSansMono.ttf',
            FONT_EMPHASIZED: 'DejaVuSansMono-Bold.ttf',
            FONT_NEWWORD: 'DejaVuSansMono-Oblique.ttf',
        })

    word_plot_info_matrix, actual_plotted_word_size_dict_list, max_scores, actual_min_x, actual_max_x = generate_plotdata(output_folder, word_info_matrix, names, new_words_lists, min_x, max_x, params)

    # The calculations whether the words collide are made from size plot data that can only be retrieved from matplotlib after the actual plotting is done. Therefore, two plots have to be made, where the sizes from the first is used in the second.
    images, cropboxes = plot_from_plotdata(output_folder, word_plot_info_matrix, names, new_words_lists, actual_plotted_word_size_dict_list, max_scores, min_x, max_x, actual_min_x, actual_max_x, params)
    if output_folder is not None:
        if params.unified_graph:
            names = ["unified"]
        for title, image in zip(names, images):
            file_name = os.path.join(output_folder, construct_pdf_file_name(title, output_folder))
            open(file_name, "wb").write(image.getbuffer())
            print("Saved plot in " + file_name)
    return images, cropboxes

def generate_clouds(corpus_folder, word2vec_model, output_folder, corpus_texts=None, **kwargs):
    params = Params(**kwargs)
    if output_folder is not None and not os.path.exists(output_folder):
        os.mkdir(output_folder)
    word_info_matrix, new_words_lists, min_x, max_x, names = generate_clouds_vectorize(corpus_folder, word2vec_model, output_folder, params, corpus_texts=corpus_texts)
    return generate_plot(output_folder, word_info_matrix, new_words_lists, min_x, max_x, names, params)
