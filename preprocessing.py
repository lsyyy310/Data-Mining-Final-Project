import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
import nltk
from nltk import stem, tokenize
from nltk.corpus import wordnet
from nltk.sentiment.vader import SentimentIntensityAnalyzer


TEXT_FEATURES = [
    "Artist", "Track", "Album", "Album_type", "Title", "Channel", "Description"
]
FEATURE_KEYWORDS = {
    "Track": [
        "feat", "remix", "vivo", "version", "remaster", 
        "remastered", "live", "dance", "edit", "radio", 
        "christmas", "single"
    ],
    "Album": [
        "edition", "deluxe", "original", "soundtrack", "feat", 
        "vivo", "remix", "version", "vol", "remastered", 
        "christmas", "anniversary", "remaster", "special", "ii"
    ],
    "Title": [
        "official", "oficial", "officiel", "feat", "ft"
        "video", "audio", "clip", "lyric", "vivo", 
        "version", "live", "remix",
    ]
}
SPECIAL_SYMBOLS = ["!", "?"]

def get_wordnet_pos(word: str):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV
    }

    return tag_dict.get(tag, wordnet.NOUN)

def tokenize_text(text: str):
    return tokenize.word_tokenize(str(text).lower())

def lemmatize_text(text: str):
    lemmatizer = stem.WordNetLemmatizer()
    words = tokenize_text(text)

    return " ".join([lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in words])

def multihot_vectorize(df, col, keywords):
    features_series = []
    
    series = df[col].fillna("").apply(lemmatize_text)
    for keyword in keywords:
        features_series.append(series.str.contains(
            keyword, case=False, na=False
        ).astype(int))
    
    merge = pd.concat(features_series, axis=1)
    merge.columns = [f"{col}_keyword_{keyword}" for keyword in keywords]
    
    return merge
    
def count_special_symbols(text: str):
    if pd.isna(text):
        return 0
    text = str(text)
    
    return sum(text.count(symbol) for symbol in SPECIAL_SYMBOLS)

def analyze_sentiment(df, col):
    sia = SentimentIntensityAnalyzer()
    sentiment_score = pd.DataFrame(
        [list(sia.polarity_scores(str(text)).values()) for text in df[col]],
        index=df.index,
        columns=[f"{col}_sentiment_{type}" for type in ["neg", "neu", "pos", "compound"]]
    )

    return sentiment_score

def create_artist_features(df):
    popularity_features = df.groupby("Artist").agg({
        "Views": "mean",
        "Likes": "mean"
    })
    style_features = df.groupby("Artist").agg({
        "Energy": "mean",
        "Valence": "mean",
        "Danceability": "mean"
    })

    merge = pd.concat([popularity_features, style_features], axis=1)
    merge.columns = [
        "Artist_avg_views", "Artist_avg_likes",
        "Artist_energy_style", "Artist_mood_style", "Artist_dance_style"
    ]
    return merge

def preprocess_text(**kwarg):
    album_type_encoder = None
    channel_encoder = None

    return_dict = {}
    for key, df in kwarg.items():
        if key == "X_train":
            # Album_type -> one hot
            album_type_encoder = OneHotEncoder(
                sparse_output=False,
                handle_unknown="ignore"
            )
            Album_type_features = album_type_encoder.fit_transform(df[["Album_type"]])
            df = df.join(pd.DataFrame(
                Album_type_features,
                index=df.index,
                columns= ["Album_type_" + name for name in album_type_encoder.categories_[0]]
            ))
            print(f"Convert Album_type to one-hot ({len(album_type_encoder.categories_[0])})")

            # Channel -> onehot, select the top30 channels
            channel_encoder = OneHotEncoder(
                categories=[list(df["Channel"].value_counts().keys())[:30]],
                sparse_output=False,
                handle_unknown="ignore"
            )
            channel_encoder.fit([[None]])
            print(f"Convert Channel to one-hot ({len(channel_encoder.categories_[0])})")

        else:
            # Album_type -> one hot
            Album_type_features = album_type_encoder.transform(df[["Album_type"]])
            df = df.join(pd.DataFrame(
                Album_type_features,
                index=df.index,
                columns= ["Album_type_" + name for name in album_type_encoder.categories_[0]]
            ))
        
        # Artist -> create popularity and style features
        artist_features = create_artist_features(df)
        df = df.merge(artist_features, on="Artist", how="left")
        

        # keywords: Track, Album, Title
        for feature_name, keywords in FEATURE_KEYWORDS.items():
            df = df.join(multihot_vectorize(df, feature_name, keywords))

        # special symbols: Track, Title
        for feature_name in ("Track", "Title"):
            df[f"{feature_name}_special_symbols"] = df[feature_name].apply(count_special_symbols)

        # length & sentiment: Track, Title, Description
        for feature_name in ("Track", "Title", "Description"):
            df[f"{feature_name}_length"] = df[feature_name].apply(
                lambda x: len(tokenize_text(x))
            )
            df = df.join(analyze_sentiment(df, feature_name))
        
        # Channel -> onehot
        channel_features = channel_encoder.transform(df[["Channel"]])
        df = df.join(pd.DataFrame(
            channel_features,
            index=df.index,
            columns= ["Channel_" + name for name in channel_encoder.categories_[0]]
        ))

        # drop
        df.drop(columns=TEXT_FEATURES, inplace=True)
        return_dict[key] = df
        
    return return_dict

def standardize(**kwarg):
    scalers = {
        "spotify": (
            StandardScaler(), 
            ["Key", "Loudness", "Tempo", "Duration_ms"]
        ),
        "yt": (
            StandardScaler(),
            ["Views", "Likes", "Comments", "Artist_avg_views", "Artist_avg_likes"]
        ),
        "length": (
            MinMaxScaler(),
            ["Track_length", "Title_length", "Description_length"]
        )
    }

    return_dict = {}
    for key, df in kwarg.items():
        for scaler_name, (scaler, features) in scalers.items():
            if key == "X_train":
                if scaler_name == "yt":
                    df[features] = scaler.fit_transform(np.log1p(df[features]))
                else:
                    df[features] = scaler.fit_transform(df[features])
            else:
                if scaler_name == "yt":
                    df[features] = scaler.transform(np.log1p(df[features]))
                else:
                    df[features] = scaler.transform(df[features])
        return_dict[key] = df
    
    return return_dict

def preprocess(**kwarg):
    return_dict = preprocess_text(**kwarg)
    return_dict = standardize(**return_dict)
    
    return return_dict