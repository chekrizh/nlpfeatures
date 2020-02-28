import pandas as pd
from nlp_primitives import LSA, PartOfSpeechCount, PolarityScore, StopwordCount, DiversityScore, MeanCharactersPerWord


def feature_engineering(df, text_column,
                        useLSA=True, usePartOfSpeechCount=True, usePolarityScore=True,
                        useStopwordsCount=True, useDiversityScore=True, useMeanCharactersPerWord=True):

    arguments = locals().copy()
    arguments = {key: val for key, val in arguments.items() if type(val) is bool}
    functions = {
        'useLSA': add_lsa,
        'usePartOfSpeechCount': add_part_of_speech_count,
        'usePolarityScore': add_polarity_score,
        'useStopwordsCount': add_stopwords_count,
        'useDiversityScore': add_diversity_score,
        'useMeanCharactersPerWord': add_mean_characters_per_word
    }
    df = df.copy()
    for item in arguments:
        functions[item](df, text_column)
    return df


def __create_columns(df, vectors, prefix):
    for i in range(len(vectors)):
        df[prefix + str(i)] = vectors[i]
    return df


def add_lsa(df, text_column):
    lsa = LSA()
    lsa_vectors = lsa(df[text_column]).tolist()
    df = __create_columns(df, lsa_vectors, 'lsa_')
    return df


def add_polarity_score(df, text_column):
    pscores = PolarityScore()
    df['polarity'] = pscores(df[text_column]).tolist()
    return df


def add_part_of_speech_count(df, text_column):
    pscount = PartOfSpeechCount()
    pscount_vectors = pscount(df[text_column]).tolist()
    df = __create_columns(df, pscount_vectors, 'psc_')
    return df


def add_stopwords_count(df, text_column):
    stop_words = StopwordCount()
    df['stop_words'] = stop_words(df[text_column]).tolist()
    return df


def add_diversity_score(df, text_column):
    unic = DiversityScore()
    df['unics'] = unic(df[text_column]).tolist()
    return df


def add_mean_characters_per_word(df, text_column):
    mean_characters_per_word = MeanCharactersPerWord()
    df['mean_characters_per_word'] = mean_characters_per_word(df[text_column]).tolist()
    return df


test = pd.DataFrame({'text': [
    "And let's say that we want to create the service that is our block for for a cube.",
    "I want to listen the ports 80 and I want to expose this out of the cluster.",
    "we're going to try and carry that forward to repeating the reporting here to the community on a bi weekly basis."
]})
res = feature_engineering(test, 'text')

print(test)
print(res)
