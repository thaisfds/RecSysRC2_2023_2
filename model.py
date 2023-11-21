import numpy as np
import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset
import sys

def construct_dataset(content, ratings, features, data):
    dataset = Dataset()
    dataset.fit((x for x in ratings['UserId'].to_list()),
                (x for x in ratings['ItemId'].to_list()))
    
    interactions, weights = dataset.build_interactions(((x['UserId'], x['ItemId'])
                                                        for x in ratings.to_dict(orient='records')))
    dataset.fit_partial(items=(x for x in content['ItemId'].to_list()),
            item_features=features)
    item_features = dataset.build_item_features(data)

    return dataset, interactions, item_features

def train_model(interactions, item_features):
    model = LightFM(loss='warp', random_state=None, max_sampled=3)
    model.fit(interactions, item_features=item_features, epochs=5)

    return model

def sort_by_features(df, content):
    predictions = pd.merge(df,content,on='ItemId')
    predictions.sort_values(by=['imdbVotesOg','Nominations','Wins','imdbRatingOg','ItemId'],ascending=[False,False,False,False,True],inplace=True)
    return predictions[['UserId','ItemId']]

#Previsões com lightfm
def sort_by_predictions(df, content, user_id_map, item_id_map, item_feat, model):
    #converter ID de usuários em mapeamento interno lightfm
    users=df['UserId'].apply(lambda u:user_id_map[u]).values
    items=df['ItemId'].apply(lambda i:item_id_map[i]).values

    #prever pontuações
    predictions = model.predict(users,items,item_features=item_feat)
    df['Prediction']=predictions

    #padronizar para permitir a combinação com item_features
    df['Prediction'] = (df['Prediction']-df['Prediction'].mean())/df['Prediction'].std()

    #atribua um peso à popularidade dos itens padronizados no dataset (já que os usuários parecem preferir itens populares)
    df = pd.merge(df,content,on='ItemId')
    df['Prediction'] +=0.25*(df['imdbVotesOg']-df['imdbVotesOg'].mean())/df['imdbVotesOg'].std()

    #retornar previsões de user-item classificadas por rating, decrescente
    df.sort_values(by=['Prediction','ItemId'],ascending=[False,True],inplace=True)
    return df[['UserId','ItemId']]

def predict(content, ratings, targets, features, data):

    dataset, interactions, item_features = construct_dataset(content, ratings, features, data)
    model = train_model(interactions, item_features)

    user_id_map, user_feature_map, item_id_map, item_feature_map = dataset.mapping()
    df_by_user={}
    #cria um conjunto de dados para as previsões de cada usuário
    for user, d in targets.groupby('UserId'):
        df_by_user[user] = d

    #prever para cada usuário
    for user in df_by_user:
        #coldstart
        if not user in user_id_map:
            df_by_user[user] = sort_by_features(df_by_user[user], content)
        #not coldstart
        else:
            df_by_user[user] = sort_by_predictions(df_by_user[user], content, user_id_map, item_id_map, item_features, model)

    #retornar previsões de user-item em userId crescente e ordem de rating decrescente
    return pd.concat(list(df_by_user.values()))




    