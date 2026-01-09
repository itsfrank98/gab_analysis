import html
import regex
import pandas as pd
import unicodedata
from bs4 import BeautifulSoup
from tqdm import tqdm
from argparse import ArgumentParser

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


def normalize_text(s: str) -> str:
    # Decompose Unicode (e.g., superscripts) into base characters
    decomposed = unicodedata.normalize("NFKD", s)
    # Keep only standard ASCII characters
    return "".join(c if ord(c) < 128 else " " for c in decomposed)


def preprocess_content(df, content_field: str, rm_punctuation=True, rm_stopwords=True, lowerize=True):
    """
    When preprocessing the content we want to:
    1. remove html tags, keeping the hashtags
    2. replace html characters such as &quot with the corresponding character
    :param df: dataframe containing the text to process
    :param content_field: name of the column containing the text
    :param rm_punctuation: if True, remove punctuation
    :param rm_stopwords: if True, remove stopwords
    :param lowerize: if True, lowercase
    :return:
    """
    dicts = []
    for i, r in tqdm(df.iterrows()):
        text_unprocessed = r[content_field]
        try:
            soup = BeautifulSoup(text_unprocessed, 'html.parser')
        except TypeError:
            print(i)
            print(r["account_id"])
            print(r["content"])
            print("\n")
        # Remove the html tags
        if lowerize:
            text = soup.get_text().lower()
        else:
            text = soup.get_text()

        # Replace html symbols like this one --> &quot; with its corresponding character
        cleaned_text = html.unescape(text)

        if rm_punctuation:
            cleaned_text = regex.sub(r"\p{P}+", "", cleaned_text)

        splitted = cleaned_text.split()
        splitted = [w for w in splitted if not any(w.__contains__(c) for c in ["^", "http", "www", "$"])]
        if rm_stopwords:
            splitted = [w for w in splitted if w not in stopwords.words("english")]
        cleaned_text = " ".join(splitted)

        r_d = dict(r)
        r_d[content_field] = cleaned_text
        dicts.append(r_d)
    preprocessed_df = pd.DataFrame(dicts)
    preprocessed_df[content_field] = preprocessed_df[content_field].apply(normalize_text)
    if "Unnamed: 0" in preprocessed_df.columns:
        preprocessed_df = preprocessed_df.drop(columns=["Unnamed: 0"])
    preprocessed_df[content_field] = preprocessed_df[content_field].replace(r'\s+', ' ', regex=True).str.strip()
    return preprocessed_df


def concat(l: pd.Series):
    l = l.tolist()
    return " ".join(l)


def concatenate_by_user(df, aggregator_column, text_column):
    ser = df.groupby(aggregator_column)[text_column].apply(concat)
    new_df = pd.DataFrame(columns=[aggregator_column, text_column])
    new_df[aggregator_column] = ser.index
    new_df[text_column] = ser.values
    return new_df


def main():
    parser = ArgumentParser()
    parser.add_argument("--input_fname", type=str)
    parser.add_argument("--output_fname", type=str)
    parser.add_argument("--content_field", type=str)
    parser.add_argument("--rm_punctuation", action="store_true")
    parser.add_argument("--rm_stopwords", action="store_true")
    parser.add_argument("--lowerize", action="store_true")
    parser.add_argument("--concat_by", type=str)
    args = parser.parse_args()

    input_fname = args.input_fname   # "dataset/comments_9_columns.h5"
    output_fname = args.output_fname     # "dataset/comments_preprocessed.csv"
    content_field = args.content_field
    rm_punctuation = args.rm_punctuation
    rm_stopwords = args.rm_stopwords
    lowerize = args.lowerize
    concat_by = args.concat_by

    if input_fname.endswith(".csv"):
        df = pd.read_csv(input_fname)
    else:
        df = pd.read_hdf(input_fname, key="df")

    preprocessed = preprocess_content(df, content_field=content_field, rm_punctuation=rm_punctuation, rm_stopwords=rm_stopwords, lowerize=lowerize)
    if concat_by:
        preprocessed = concatenate_by_user(preprocessed, aggregator_column=concat_by, text_column=content_field)
    preprocessed = preprocessed.dropna(subset=content_field)
    preprocessed.to_csv(output_fname, encoding="utf-8")


if __name__ == "__main__":
    main()
