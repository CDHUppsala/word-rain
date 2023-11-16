## Word Rain
Code for generating word rains (= a semantically aware version of the word cloud visualisation of texts)

Word Rain is a collaboration between, [CDHU](https://www.abm.uu.se/cdhu-eng/), the [National Language Bank of Sweden/CLARIN Knowledge Centre for the Languages of Sweden](https://www.isof.se/other-languages/english/clarin-knowledge-centre-for-the-languages-of-sweden-swelang) at the Language Council of Sweden and the [iVis group](https://ivis.itn.liu.se) at Linköping University.

Example of a word rain IPCC report Climate Change 2021: The Physical Science Basis, Summary for Policymakers, Summary for Policymakers.
![Example of a word rain](word-rain-example.png)


### Installations
For a conda environment, the following is needed

conda install numpy

conda install scipy

conda install scikit-learn

conda install gensim

conda install nltk

conda install matplotlib

conda install -c conda-forge python-bidi

### Running 
See the `visualise_climate_texts.py` for an example of how the code in `wordrain/visualise_climate_texts.py` is to be used.

You need a word2vec space, a stop word list and a corpus containing several sub-corpora. The folder CORPUS_FOLDER should contain sub-folders, which each of them contain the sub-corpora to be visualised. The sub-folders, in turn, contain (one or several) .txt-files, where the corpus is stored. The folder "swedish-climate-texts", gives an example of this structure. The Swedish word2vec space used in the example can be found here: http://vectors.nlpl.eu/repository/

```
WORD_SPACE_PATH = "../../../wordspaces/69/model.bin"
STOP_WORDS_LIST = "swedish_stopwords_climate.txt"
CORPUS_FOLDER = "swedish-climate-texts"
OUTPUT_FOLDER = "images_climate"
```


