## Word Rain
Code for generating word rains (= a semantically aware version of the word cloud visualisation of texts)

Word Rain is a collaboration between [Centre for Digital Humantieis and Social Sciences, Uppsala (CDHU)](https://www.abm.uu.se/cdhu-eng/), the [National Language Bank of Sweden/CLARIN Knowledge Centre for the Languages of Sweden](https://www.isof.se/other-languages/english/clarin-knowledge-centre-for-the-languages-of-sweden-swelang) at the Language Council of Sweden and the [iVis group](https://ivis.itn.liu.se) at Linköping University.

Example of a word rain of the IPCC report: 'Climate Change 2021: The Physical Science Basis, Summary for Policymakers'.
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
See `visualise_climate_texts.py` for an example of how the code in `wordrain/generate_wordrain.py` is to be used.

You need a word2vec space, a stop word list and a corpus containing several sub-corpora. The folder CORPUS_FOLDER should contain sub-folders, which each of them contain the sub-corpora to be visualised. The sub-folders, in turn, contain (one or several) .txt-files, where the corpus is stored. The folder "swedish-climate-texts", gives an example of this structure. The Swedish word2vec space used in the example can be found here: http://vectors.nlpl.eu/repository/

```
WORD_SPACE_PATH = "../../../wordspaces/69/model.bin"
STOP_WORDS_LIST = "swedish_stopwords_climate.txt"
CORPUS_FOLDER = "swedish-climate-texts"
OUTPUT_FOLDER = "images_climate"
```

## Presentations
- Skeppstedt, M., Aangenendt, G. and Söderfeldt, Y. ‘Visualizing longitudinal trends in digitized periodicals from the Swedish diabetes association’, Huminfra Conference, Gothenburg, Sweden, 2024. [pdf](https://demo.spraakbanken.gu.se/gerlof/abstract_submissions.pdf)
- Ahltorp, M. and Hessel, J. and Eriksson, G. and Skeppstedt, M. (2023) “Visualisering av ett lexikons täckning av olika textgenrer”, NFL2023: 17. konferansen om leksikografi i Norden, Bergen, 24.-26. May 2023.[pdf](https://www.uib.no/sites/w3.uib.no/files/attachments/samandrag-nfl-17.pdf#page=11)
- Skeppstedt, M. and Ahltorp, M. (2023). “The Words of Climate Change: TF-IDF-based word clouds derived from climate change reports”.  Digital Humanities in the Nordic and Baltic Countries 2023. Digital/Universities of Oslo, Bergen and Stavanger, Norway. 

## Acknowledgements
The work on Word Rain is partly supported by three research infrastructures:

- [Huminfra](https://www.huminfra.se): National infrastructure for Research in the Humanities and Social Sciences (Swedish Research Council, 2021-00176)
- [InfraVis](https://infravis.se): the Swedish National Research Infrastructure for Data Visualization (Swedish Research Council, 2021-00181)
- [Nationella Språkbanken](https://www.språkbanken.se/sprakbankeninenglish.html): The National Language Bank of Sweden (Swedish Research Council, 2017-00626)
