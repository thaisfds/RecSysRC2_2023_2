import pandas as pd
import numpy as np
import re

def generate_genres(df, column):
    #Separa os generos
    #Substitui ' ' por ''
    genres = df[column].str.replace(' ','').str.split(',')

    #Seleciona os generos unicos
    unique_genres = set()

    for row in genres:
        for element in row:
            unique_genres.add(element)

    #Cria duas novas colunas para marcar se o conteudo tem ou nao o genero
    features_vector = []

    for genre in unique_genres:
        df[genre] = df[column].apply(lambda y: genre in y)
        features_vector.append(genre+':False')
        features_vector.append(genre+':True')

    #Adapta as features para o modelo
    #Se um filme pertence aos gêneros “Ação” e “Aventura”, a lista de strings correspondente terá os elementos “Ação:True” e “Aventura:True”.
    features_matrix = []

    for i in df.to_dict(orient='Records'):
        features_matrix.append([g+':'+str(i[g]) for g in unique_genres])

    return features_vector, features_matrix


def convert_str(value):
    if value !='N/A':
        value=value.replace(',','')
        return str(round(float(value)))
    else:
        return ''


def str_to_int(df, column):
    df[column] = df[column].apply(convert_str)
    unique_values = df[column].drop_duplicates().tolist()
    formatted_values = [f'{column}:{value}' for value in unique_values]
    return formatted_values

def categorize_votes(df, column):
    df.loc[df[column] > 10**5, column] = 10**5
    df.loc[(df[column] < 10**5) &(df[column] > 10**4),column] = 10**4
    df.loc[(df[column] < 10**4) &(df[column] > 10**3),column] = 10**3
    df.loc[(df[column] < 10**3) &(df[column] > 10**2),column] = 10**2
    df.loc[(df[column] < 10**2),column] = 10**1
    df.loc[df[column].isna(),column] ='N/A'
    df[column]=df[column].astype(str)
  
def categorize_awards(df, column, prefix):
    awards = df[column].copy()
    awards[(awards>=1) & (awards<5)] = 1
    awards[(awards>=5) & (awards<10)] = 5
    awards[(awards>=10) & (awards<15)] = 10
    awards[(awards>=15)] = 15
    features = [prefix+str(award) for award in awards]
    features_unique = [prefix+'0', prefix+'1', prefix+'5',prefix+'10', prefix+'15']
    return features_unique, features

def copy_column(df, column):
    column_og = column+'Og'
    df[column_og] = df[column].copy()


def clean_lang(string):
    # Remove leading and trailing spaces, split by comma, and take the first part
    cleaned_string = string.split(',')[0].strip()
    return cleaned_string

def preprocess_lang(df, column, filter):
    # Apply the cleaning function to the column column
    df[column] = df[column].apply(clean_lang)

    # Replace 'N/A' with 'None'
    df.loc[df[column] == 'N/A', column] = 'None'

    # Replace values with count less than 200 with 'Other'
    string_counts = df[column].value_counts()
    low_count_strings = df[column].map(string_counts) < filter
    df.loc[low_count_strings, column] = 'Other'

def extract_awards(awards, suffix):
    # Use regular expression to extract numerical nominations
    results = re.findall(rf'([0-9]+) {suffix}', awards)
    
    # Join multiple nominations with a comma
    return ','.join(results)

def clean_and_convert_awards(df, column, suffix):
    # Extract nominations and replace non-numeric characters
    df[column] = df['Awards'].apply(lambda y: extract_awards(y, suffix))
    df[column] = df[column].replace(r'[a-zA-Z]+', '', regex=True)
    
    # Replace empty strings with 0
    df.loc[df[column] == '', column] = '0'
    
    # Convert column column to numeric
    df[column] = pd.to_numeric(df[column])


def preprocess_features(df):
    # Generate original versions of columns
    copy_column(df, 'imdbVotes')
    copy_column(df, 'imdbRating')

    # Convert votes to int type
    str_to_int(df,'imdbVotesOg')
    df['imdbVotesOg']=pd.to_numeric(df['imdbVotesOg'])

    # Language and country features
    # Filter languages and countries that appear less than 200 times
    preprocess_lang(df, 'Language', 200)
    preprocess_lang(df, 'Country', 200)
    language_features = list(df['Language'].unique())
    country_features = list(df['Country'].unique())

    # Generate nominations and wins columns
    clean_and_convert_awards(df, 'Nominations', 'nomination')
    clean_and_convert_awards(df, 'Wins', 'win')

    # Generate columns for each genre
    genre_features, genre_feats = generate_genres(df, 'Genre')
    
    # Categorize nominations and wins
    nom_features, nom_feats = categorize_awards(df,'Nominations','n:')
    wins_features, wins_feats = categorize_awards(df,'Wins','w:')

    # Ratings features
    rating_features = str_to_int(df,'imdbRating')

    # Votes features
    str_to_int(df,'imdbVotes')
    df['imdbVotes']=pd.to_numeric(df['imdbVotes'])
    categorize_votes(df,'imdbVotes')
    votes_features = str_to_int(df,'imdbVotes')

    # Create data for model
    item_ids = df['ItemId'].to_list()
    data = [(item_ids[i],g) for i,g in enumerate(genre_feats)]
    data = [(data[i][0],data[i][1]+['imdbRating:'+j]) for i,j in enumerate(df['imdbRating'])]
    data = [(data[i][0],data[i][1]+['imdbVotes:'+j]) for i,j in enumerate(df['imdbVotes'])]
    data = [(data[i][0],data[i][1]+[w]) for i,w in enumerate(wins_feats)]
    data = [(data[i][0],data[i][1]+[w]) for i,w in enumerate(nom_feats)]
    data = [(data[i][0],data[i][1]+[w]) for i,w in enumerate(df['Language'])]
    data = [(data[i][0],data[i][1]+[w]) for i,w in enumerate(df['Country'])]

    # Create list of features for model
    features = genre_features + rating_features + votes_features + wins_features + language_features + country_features + nom_features

    return features, data
