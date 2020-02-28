# NLP features: user guide
 
This library is a wrapper over the nlp primitives library. 
Allows you to quickly pull out the maximum number of features from the text.

#How to install
```
pip install nlpfeatures
```
*The process may take some time due to the large number of dependencies that are required for features engineering*
# Quick start 
```
from nlpfeatures import feature_engineering
import pandas as pd


df = pd.DataFrame({'text': [
    "And let's say that we want to create the service that is our block for for a cube.",
    "I want to listen the ports 80 and I want to expose this out of the cluster.",
    "we're going to try and carry that forward to repeating the reporting here to the community on a bi weekly basis."
]})
res = feature_engineering(df, 'text')
```
# feature_engineering()
Takes an input object and the name of the column containing the text. Returns a copy of the data frame with new columns.
 You can control which features will be created.  
 __Parameters:__  
 * __df__: object DataFrame from Pandas
 * __text_column__: str, name of column in df
 * __useLSA__: boolean, optional (default=True)  
 * __usePartOfSpeechCount__: boolean, optional (default=True)
 * __usePolarityScore__ : boolean, optional (default=True)
 * __useStopwordsCount__: boolean, optional (default=True)
 * __useDiversityScore__: boolean, optional (default=True)
 * __useMeanCharactersPerWord__: boolean, optional (default=True)

 #About features
 *All features have their own function, to add only their own columns*
 
 ## LSA
Calculates the Latent Semantic Analysis Values of Text Input

Description:  
Given a column of strings, transforms those strings using tf-idf and single value decomposition to go from a sparse matrix to a compact matrix with two values for each string.
These values represent that Latent Semantic Analysis of each string. These values will represent their context with respect to [nltk’s brown sentence corpus.](https://www.nltk.org/book/ch02.html#brown-corpus)

__Solo function__: add_lsa(df, text_column)

## PartOfSpeechCount
Calculates the occurences of each different part of speech.

Description:  
Given a column of strings, categorize each word in the string as a different part of speech, and return the total count for each of 15 different categories of speech.

__Solo function__: add_part_of_speech_count(df, text_column)

## PolarityScore
Calculates the polarity of a text on a scale from -1 (negative) to 1 (positive)

Description:  
Given a column of strings assign a polarity score from -1 (negative text), to 0 (neutral text), to 1 (positive text). The functions returns a score for every given piece of text.

__Solo function__: add_polarity_score(df, text_column)

## StopwordsCount
Determines number of stopwords in a string.

Description:  
Given column of strings, determine the number of stopwords characters in each string. Looks for any of the English stopwords defined in nltk.corpus.stopwords. Case insensitive

__Solo function__: add_stopwords_count(df, text_column)

## DiversityScore
Calculates the overall complexity of the text based on the total number of words used in the text

Description:  
Given a column of strings, calculates the total number of unique words divided by the total number of words in order to give the text a score from 0-1 that indicates how unique the words used in it are. This primitive only evaluates the ‘clean’ versions of strings, so ignoring cases, punctuation, and stopwords in its evaluation.

__Solo function__: add_diversity_score(df, text_column)

## MeanCharactersPerWord
Determines the mean number of characters per word.

Description:  
Given column of strings, determine the mean number of characters per word in each string. A word is defined as a series of any characters not separated by white space. Punctuation is removed before counting.
