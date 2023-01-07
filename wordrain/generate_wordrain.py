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

NR_TEXTS_TO_INCLUDE_FROM_BACKGROUND = 10000
NR_OF_WORDS_TO_SHOW = 100
FONTSIZE = 16
X_STRETCH = 30 # How much to make the x-coordinates wider
X_LENGTH = 10 # Default value for determine if two texts colide x-wise (typically changed for different corpora, then the default is not used)
Y_LENGTH = 10 # Determines if two texts colide y-wise, and how much to move, when the texts collide
DEBUG = False

matplotlib.rcParams["font.family"] = "monospace"

# Very simple compound spitting, not really language independent, but also not very well adapted to any language
def get_compound_vector(word, word2vec_model):
    return_vector = None
    for i in range(4, len(word) - 3):
        first = word[:i].lower()
        second = word[i:].lower()
        if first in word2vec_model and second in word2vec_model:
            first_v = word2vec_model[first]
            second_v = word2vec_model[second]
            return_vector = np.add(first_v, second_v)
            return return_vector
                
    for j in range(len(word)-1, 4, -1):
        first_alone = word[:j].lower()
        if first_alone in word2vec_model:
            return_vector = word2vec_model[first_alone]
            return return_vector
        second_alone = word[-j:].lower()
        if second_alone in word2vec_model:
            return_vector = word2vec_model[second_alone]
            return return_vector
            
    return return_vector
        

def get_vector_for_word(word, word2vec_model):
    if word in word2vec_model:
        vec_raw  = word2vec_model[word]
        norm_vector = list(preprocessing.normalize(np.reshape(vec_raw, newshape = (1, 100)), norm='l2')[0])
        return norm_vector
        
    if " " in word:
        sub_words = word.split(" ")
        acc_vector = None
        for sw in sub_words:
            if sw in word2vec_model:
                vec_raw  = word2vec_model[sw]
                acc_vector = vec_raw # This will lead to that only the last in a +1-gram
                # will contribute to the vector. But otherwise the tsne space puts to much
                # emphasis on the +1-grams
                """
                if type(acc_vector) == np.ndarray:
                    acc_vector = np.add(vec_raw, acc_vector)
                else:
                    acc_vector = vec_raw
                """
        if type(acc_vector) != np.ndarray:
            compound_vector = get_compound_vector(word, word2vec_model)
            if type(compound_vector) == np.ndarray:
                acc_vector = compound_vector
                
        if type(acc_vector) == np.ndarray:
            norm_vector = list(preprocessing.normalize(np.reshape(acc_vector, newshape = (1, 100)), norm='l2')[0])
            return norm_vector
            
    compound_vector = get_compound_vector(word, word2vec_model)
    if type(compound_vector) == np.ndarray:
        norm_vector = list(preprocessing.normalize(np.reshape(compound_vector, newshape = (1, 100)), norm='l2')[0])
        return norm_vector
    
    print("No word2vec vector found for:", word)
    return None


def is_point_in_bounding_box(point_x, point_y,\
    prev_min_point_x, prev_max_point_x, prev_min_point_y, prev_max_point_y):
    return (point_x >= prev_min_point_x and \
                point_x <= prev_max_point_x and \
                point_y >= prev_min_point_y and \
                point_y <= prev_max_point_y)
                


def do_plot(word_vec_dict, word_list, sorted_word_scores_vec, picname, title, nr_of_words_to_show, x_length_factor, times_found_dict, nr_of_texts, max_score, extreme_values_tuple = None, extra = "", display_log = False):

    
    if extreme_values_tuple:
        if DEBUG:
            extreme_marker_color = "red"
        else:
            extreme_marker_color = (0.0, 0.0, 0.0, 0.0)
            
        global_point_x_overall_min, global_point_x_overall_max, \
                global_point_y_overall_min, global_point_y_overall_max, global_point_x_overall_max_with_text = extreme_values_tuple
        plt.scatter(global_point_x_overall_min - x_length_factor*FONTSIZE, global_point_y_overall_min, zorder = -100000,  color = extreme_marker_color, marker = ".")
        plt.scatter(global_point_x_overall_max_with_text, global_point_y_overall_max, zorder = -100000,  color = extreme_marker_color, marker = ".")
             


    point_x_overall_min = math.inf
    point_x_overall_max = -math.inf
    point_x_overall_max_with_text = -math.inf
    point_y_overall_min = math.inf
    point_y_overall_max = -math.inf

    cmap = plt.get_cmap('cool')
    
    plt.axis('off')
 
    texts = []
    z_order_text = 0
    word_nr = 1
    #fs = FONTSIZE
        
    previous_points = []
    for score, word in sorted_word_scores_vec[:nr_of_words_to_show]:
        fs = FONTSIZE*score/max_score
        
        if display_log:
            fs = FONTSIZE*(math.log(score/max_score*100)/10)
        if word in word_vec_dict:
            word_to_display = word.replace(" ", "_")
            
            point_x = word_vec_dict[word][0]
            point_y = word_vec_dict[word][1]
            
            point_x = point_x * X_STRETCH
            #fontsize_to_use = max(fs, 1) #- 0.4*len(word_to_display)
            fontsize_to_use = fs - 0.2*len(word_to_display)
            if fontsize_to_use < 0:
                fontsize_to_use = fs
            
            point_x_before = point_x
            point_y_before = point_y
            changed = True
                
            # Use original font size, not word length adapted
            max_point_x = point_x + x_length_factor*fs*len(word_to_display)
            
            # Not necessary, just to clarify
            right_upper_max_point_x = max_point_x
            right_lower_max_point_x = max_point_x
            left_upper_max_point_x = point_x
                                
            while changed:
            
                point_y_before_loop = point_y
                for prev_point in previous_points:
                        
                    prev_min_point_x = prev_point["min"][0]
                    prev_min_point_y = prev_point["min"][1]
                    prev_max_point_x = prev_point["max"][0]
                    prev_max_point_y = prev_point["max"][1]
                        
                    
                    
                    right_upper_max_point_y = point_y + Y_LENGTH*fontsize_to_use
                    left_upper_max_point_y = point_y + Y_LENGTH*fontsize_to_use
                    right_lower_max_point_y = point_y
                
                    # Check if new point is in bounding box of prev
                    #################################################
                    # in points_to_check:
                    # left lower
                    # right upper
                    # left upper
                    # right lower
                    # middle
                        
                    points_to_check = [(point_x, point_y,),\
                        (right_upper_max_point_x, right_upper_max_point_y),\
                        (left_upper_max_point_x, left_upper_max_point_y),\
                        (right_lower_max_point_x, (right_upper_max_point_y + right_lower_max_point_y)/2),\
                        ((left_upper_max_point_x + right_lower_max_point_x)/2,\
                        right_lower_max_point_y)]
                        
                    for (xi, yi) in points_to_check:
                        if is_point_in_bounding_box(xi, yi, \
                                                    prev_min_point_x, \
                                                    prev_max_point_x, \
                                                    prev_min_point_y, \
                                                    prev_max_point_y):
                            point_y = prev_max_point_y
                            break
                    
                    # Check if prev is in bounding box of new
                    #########################################
                    # content in bounding_boxes:
                    # left lower
                    # right upper
                    # left upper
                    # right lower
                    bounding_boxes = [(point_x, \
                                            right_upper_max_point_x, \
                                            point_y, \
                                            right_upper_max_point_y),\
                                            \
                                            (point_x, \
                                            right_upper_max_point_x, \
                                            point_y, \
                                            right_upper_max_point_y),\
                                            \
                                            (point_x, \
                                            right_upper_max_point_x, \
                                            point_y,\
                                            right_upper_max_point_y),\
                                            \
                                            (point_x, \
                                            right_lower_max_point_x, \
                                            point_y, \
                                            right_upper_max_point_y),\
                                            \
                                            (point_x, \
                                            right_lower_max_point_x, \
                                            point_y, \
                                            right_upper_max_point_y), \
                                            \
                                            
                                            ]
                        
                    for (a, b, c, d) in bounding_boxes:
                        if is_point_in_bounding_box(prev_min_point_x, prev_min_point_y, \
                                a, b, c, d):
                            point_y = prev_max_point_y
                            break
                        
                    # middle
                    if ((prev_max_point_x + prev_min_point_x)/2 >= point_x and \
                            (prev_max_point_x + prev_min_point_x)/2 <= right_lower_max_point_x and \
                            (prev_max_point_y + prev_min_point_y)/2 >= point_y and \
                            (prev_max_point_y + prev_min_point_y)/2 <= right_upper_max_point_y):
                        point_y = prev_max_point_y
                        
                if point_y_before_loop == point_y:
                    changed = False
   
            min_point = (point_x, point_y)
            
            # Use original fontsize, not word length adapted
            max_point = (max_point_x, point_y + Y_LENGTH*fs)
            previous_points.append({"min": min_point, "max": max_point})
   
                         
            y_bar_start = Y_LENGTH*FONTSIZE/2 # TODO: Make y_bar_start corpus independent
            cmap_value = ((point_x/X_STRETCH)/100 + 1)/2 # Not used when actually plotting
            if extreme_values_tuple:
                cmap_value = (point_x - global_point_x_overall_min)/(global_point_x_overall_max - global_point_x_overall_min)
            color_lighter = (cmap(cmap_value)[0], cmap(cmap_value)[1], cmap(cmap_value)[2], min(0.15*fontsize_to_use, 1))
            color_lighter_lighter = (cmap(cmap_value)[0], cmap(cmap_value)[1], cmap(cmap_value)[2], min(0.1*fontsize_to_use, 1))
            
            
            # The bar upwards
            # TODO: Make the length of the bar (including the log version) more corpus-independent
            y_bar_length = score/max_score*FONTSIZE*Y_LENGTH*5#*nr_of_words_to_show/100
            if display_log:
                y_bar_length = math.log(y_bar_length, 1.02)/1.5
            upwardbar_end = y_bar_start + y_bar_length
            
            linewidth=score/max_score
            if display_log:
                linewidth = math.log(score/max_score*100)/10
                
            plt.plot([point_x, point_x], [y_bar_start, upwardbar_end], color=cmap(cmap_value), zorder = -1000-z_order_text, linewidth=linewidth)
            plt.scatter(point_x, upwardbar_end, zorder = -100000, color=cmap(cmap_value), marker = "o", s=fontsize_to_use*0.4*score/max_score)
            
            
            # The texts have been changed to be printed downwards instead
            negative_text_y = -point_y
            
            #plt.plot([point_x, max_point_x], [-point_y, -point_y + Y_LENGTH*fontsize_to_use], color="red", linewidth = 0.1)
            if DEBUG:
                plt.plot([point_x, max_point_x], [negative_text_y, -max_point[1]], color="red", linewidth = 0.1)

            
            # The text
            if times_found_dict[word] == nr_of_texts:
                style = "normal" # found in all texts
            else:
                style = "italic"
            if times_found_dict[word] == 1: # only in this text
                fontweight = "bold"
            else:
                fontweight = "normal"
                
            #text_color = (0.0, 0.0, 0.0, min(1.0, 2*score/max_score))
            text_color = (0.0, 0.0, 0.0)
            t = plt.text(point_x, negative_text_y, word_to_display, zorder = z_order_text, color = text_color, fontsize=fontsize_to_use, fontstyle=style, fontweight=fontweight, verticalalignment="top")
            
            
            # The bar downwards
            plt.plot([point_x, point_x], [y_bar_start, negative_text_y], color=color_lighter, linewidth=0.1*score/max_score, zorder = -10000000)
            plt.scatter(point_x, negative_text_y, zorder = -100000, color=color_lighter_lighter, marker = "o", s=fontsize_to_use*0.05)
            
            z_order_text =  z_order_text - 1
            #fs = fs - 0.6*score/max_score
            
            min_y_used = negative_text_y - Y_LENGTH*fontsize_to_use
            max_y_used = upwardbar_end + fontsize_to_use
            if min_y_used < point_y_overall_min:
                point_y_overall_min = min_y_used
            if max_y_used > point_y_overall_max:
                point_y_overall_max = max_y_used
                
            max_x_used =  point_x
            min_x_used = point_x
            max_x_used_with_text = max_x_used + fontsize_to_use*len(word_to_display)*x_length_factor
            if min_x_used < point_x_overall_min:
                point_x_overall_min = min_x_used
            if max_x_used > point_x_overall_max:
                point_x_overall_max = max_x_used
            if max_x_used_with_text > point_x_overall_max_with_text:
                point_x_overall_max_with_text = max_x_used_with_text
            
        else:
            print("Word not found, strange", word)
        word_nr = word_nr + 1

    title_to_use = "\n" + title + extra
    if picname != "":
        title_to_use = title + ", " + picname
    title_fontsize=FONTSIZE
    if display_log:
        title_fontsize=math.log(100*FONTSIZE) # TODO: Make this more general
    plt.title(title_to_use, fontsize=title_fontsize)

    return (point_x_overall_min, point_x_overall_max, point_y_overall_min, point_y_overall_max, point_x_overall_max_with_text)


def vectorize_and_plot(background_corpus, texts, names, stopwords, word2vec_model, picname, ngram_range, nr_of_words_to_show, output_folder, x_length_factor, idf, extra_words, display_log):
    
    if idf:
        extra = ""
    else:
        extra = "-tf-only"
            
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    TEXTS_FOLDER = "texts"
    texts_output = os.path.join(output_folder, TEXTS_FOLDER)
    if not os.path.exists(texts_output):
        os.mkdir(texts_output)
    
    nr_of_texts = len(texts)
    
    # For tf-idf
    ############
    vectorizer = TfidfVectorizer(stop_words=stopwords, min_df=2, smooth_idf=False, sublinear_tf=False, ngram_range = ngram_range, use_idf=idf, norm=None)
    vectorizer.fit_transform(texts + background_corpus)
    X = vectorizer.transform(texts)
    inversed = vectorizer.inverse_transform(X)
    
    
    sorted_word_scores_matrix = []
    all_words = []
    for el, w, name in zip(inversed, X, names):
        score_vec = w.toarray()[0]
        word_scord_vec = []
        for word in el:
            all_words.append(word)
            word_scord_vec.append((score_vec[vectorizer.vocabulary_[word]], word))
        word_scord_vec = sorted(word_scord_vec, reverse=True)
        sorted_word_scores_matrix.append(word_scord_vec)
        
    
        
        
    # For tsne
    ##########
    vectorizer_for_tsne = TfidfVectorizer(stop_words=stopwords, smooth_idf=False, sublinear_tf=False, ngram_range = ngram_range, use_idf=idf)
    X_for_tsne = vectorizer_for_tsne.fit_transform(texts)
    inversed_for_tsne = vectorizer_for_tsne.inverse_transform(X_for_tsne)
    
    # Create space (and score statistics):
    word_to_use_set = set()
    times_found_dict = {}
    max_scores = []
    
    for sws, title in zip(sorted_word_scores_matrix, names):
        # Write scores to file
        max_score_subcorpus = -math.inf
        tf_name = os.path.join(texts_output, title + extra + ".txt")
        with open(tf_name, "w") as tf_write:
            for s, w in sws[:nr_of_words_to_show]:
                tf_write.write(w + "\t" + str(s) + "\n")
                word_to_use_set.add(w)
                if w not in times_found_dict:
                    times_found_dict[w] = 1
                else:
                    times_found_dict[w] = times_found_dict[w] + 1
                
                if s > max_score_subcorpus:
                    max_score_subcorpus = s
        max_scores.append(max_score_subcorpus)
    
    all_vectors_list = []
    found_words = []
    not_found_words = []

    words_to_generate_space_for = set()
    for el in inversed_for_tsne:
        for word in el:
            if word in word_to_use_set:
                words_to_generate_space_for.add(word)
    for word in extra_words:
        words_to_generate_space_for.add(word)
        
    words_to_generate_space_for = sorted(list(words_to_generate_space_for))
    print(len(words_to_generate_space_for))
    
    for word in words_to_generate_space_for:
        # Add vectors
        norm_vector = get_vector_for_word(word, word2vec_model)
        if norm_vector == None:
            not_found_words.append(word)
        else:
            found_words.append(word)
            all_vectors_list.append(norm_vector)
                    
   
    all_vectors_np = np.array(all_vectors_list)
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
        point = [point[0], 0]
        word_vec_dict[found_word] = point
        if point[0] < min_x:
            min_x = point[0]
        if point[0] > max_x:
            max_x = point[0]
        if point[1] < min_y:
            min_y = point[1]
        if point[1] > max_y:
            max_y = point[1]


    # Just add a random point for words not found in the space
    for word in not_found_words:
        random_key = (random.sample(list(word_vec_dict), 1))[0]
        point = [el + 0.01*random.randint(1,10) for el in word_vec_dict[random_key]]
        word_vec_dict[word] = point
        
    global_point_x_overall_min = math.inf
    global_point_x_overall_max = -math.inf
    global_point_x_overall_max_with_text = -math.inf
    global_point_y_overall_min = math.inf
    global_point_y_overall_max = -math.inf
    
    # Make one first run, just to get the global max and min point (but don't save the result)
    for inverse, sorted_word_scores, title, max_score in zip(inversed, sorted_word_scores_matrix, names, max_scores):
        main_fig = plt.figure()
        plt.axis('off')
        point_x_overall_min, point_x_overall_max, point_y_overall_min, point_y_overall_max, point_x_overall_max_with_text = \
            do_plot(word_vec_dict, inverse, sorted_word_scores, picname, title, nr_of_words_to_show, x_length_factor, times_found_dict, nr_of_texts, max_score, extra=extra, display_log = display_log)
            
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
            
        plt.close('all')
            
    extreme_values_tuple = (global_point_x_overall_min, global_point_x_overall_max, \
                global_point_y_overall_min, global_point_y_overall_max, global_point_x_overall_max_with_text)
                
    # Make one another run and save the results in images
    for inverse, sorted_word_scores, title, max_score in zip(inversed, sorted_word_scores_matrix, names, max_scores):
        main_fig = plt.figure()
        plt.axis('off')
        
        do_plot(word_vec_dict, inverse, sorted_word_scores, picname, title, nr_of_words_to_show, x_length_factor, times_found_dict, nr_of_texts, max_score, extreme_values_tuple = extreme_values_tuple, extra = extra, display_log = display_log)
 
        file_name = os.path.join(output_folder, title.replace(" ", "-"))
        if picname != "":
            file_name = file_name + "-" + picname.replace(" ", "-")
        file_name = file_name + extra + ".pdf"
        
        plt.tight_layout()
        plt.savefig(file_name, orientation = "landscape", format="pdf", bbox_inches='tight', pad_inches=0)
        print("Saved plot in " + file_name)
        plt.close('all')
    
   
####
# The main function for generating the clouds
####
def generate_clouds(corpus_folder, word2vec_model, output_folder, x_length_factor = X_LENGTH, stopwords=[], background_corpus=[], run_data = [("1-gram", (1, 1), NR_OF_WORDS_TO_SHOW), ("2-gram", (2,2), NR_OF_WORDS_TO_SHOW)], pre_process_method = None, idf = True, extra_words_path = None, display_log = False):

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
  
    folders = glob.glob(os.path.join(corpus_folder, "*", ""))
    if len(folders) == 0:
        print("no subfolders found in ", corpus_folder)
        exit()
        
    for folder in folders:
        print("Reading files from: ", folder)
        folder_name = unicodedata.normalize("NFC", os.path.basename(os.path.split(folder)[0]))
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
   
    # Visualise the texts
    for (n, ngram_range, nr_of_words_to_show) in run_data:
        vectorize_and_plot(background_corpus, texts, names, stopwords, word2vec_model, n, ngram_range, nr_of_words_to_show, output_folder, x_length_factor, idf, extra_words, display_log)
