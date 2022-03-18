# Aspect-Based-Sentiment-Analysis_Undergraduate
Undergraduate thesis for Aspect-Based Sentiment Analysis

## Data

* 2015-2019 Zhejiang films under data/ directory

## Structure

* dataprocess.py: Preprocess the data
* word2vec.py: Generate word vectors, character vectors
* feature.py: Generate SVD features
* GCAE_word_char.py: model
* SynAtt_expand_model.py: model
* train.py: Train
* utils.py: Common functions

## Aspect

Utilitize LightGBM for classification，get TopK Aspect from feature importance. Some of features are as follows:
```
subjects = ['Theme Value Intention Feeling',
            'Script Narration Plot Structure',
            'Protagonist Actor Role',
            'Picture Photography Landscape Atmosphere',
            'Texture Background']
```

## Model

### LSTM model based on attention mechanism

* Utilize R-NET and take the word vector encoded by Word2vec pre-training model for joint representation
* Proposed a coding model represented by aspects
* Encoded the aspects and content of film reviews together
* Encoded the syntactic structure 
* Predict 5 aspects together

### Result
Precision = 80.92 
Recall = 82.66 
F1 = 81.22

