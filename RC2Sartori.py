import numpy as np
import pandas as pd
from lightfm import LightFM
import lightfm.cross_validation
from lightfm.evaluation import precision_at_k
from lightfm.data import Dataset
import sys

#read sys arguments
rating_file = sys.argv[1] if len(sys.argv >1) else "ratings.jsonl"
content_file = sys.argv[2] if len(sys.argv >2) else "content.jsonl" 
target_file = sys.argv[3] if len(sys.argv >3) else "targets.csv"  


#apply one hot encoding to a column
#only used for genres
#returns the possible features for light fm and the name of the resulting columns
def oneHotEncoding(item_features,column):
    genre = item_features[column].str.replace(' ','').str.split(',')
    unique = set()
    for x in genre:
        for j in x:
            unique.add(j)
    features=[]
    for g in unique:
        item_features[g] = item_features[column].apply(lambda s: g in s)
        features.append(g+':False')
        features.append(g+':True')
   
    return features,unique

#generate lightfm features for one hot encoded columns for all items
def genFeatures(item_features,columns):
    feat=[]
    for x in item_features.to_dict(orient='Records'):
        feat.append([g+':'+str(x[g]) for g in columns])
    return feat


#filters a column, changing its format to a column of list (instead of column of strings)
#returns lightfm possible features
def filterColumn(item_features,column,prefix='g:'):
    item_features[column] = item_features[column].str.replace(' ','').str.split(',')
    unique = set()
    for x in item_features[column]:
        for j in x:
            unique.add(j)
    features=[]
    for g in unique:
        features.append(prefix+g)
   
    return features,unique

#generates lightfm features for a column for all items
def genColumn(item_features,column,prefix='g:'):
    feat=[]
    for x in item_features.to_dict(orient='Records'):
        feat.append([prefix+str(g) for g in x[column]])
    return feat

#transform awards (nominations or wins) in categorical features based on arbitrary thresholds
def awardsFeats(item_features,column,prefix='n:'):
    awards = item_features[column].copy()
    awards[(awards>=1) & (awards<5)] = 1
    awards[(awards>=5) & (awards<10)] = 5
    awards[(awards>=10) & (awards<15)] = 10
    awards[(awards>=15)] = 15
    feats = [prefix+str(x) for x in awards]
    features = [prefix+'0',prefix+'1',prefix+'5',prefix+'10',prefix+'15']
    return features,feats

#converts number in "human" string form (e.g 2,567,000) to int in string form
def intOrNa(value):
    value=value.replace(',','')
    if value !='N/A':
        return str(round(float(value)))
    else:
        return ''

#apply intOrNa to all values in a column and return possible features for lightfm
def strToInt(item_features,column):
    item_features[column] = item_features[column].apply(lambda rating: intOrNa(rating))
    return [column+':'+str(rating) for rating in item_features[column].drop_duplicates().to_list()]

#transforms numeric column in categorical
def filterBig(df,column):
    df.loc[df[column] > 10**5,column] = 10**5
    df.loc[(df[column] < 10**5) &(df[column] > 10**4),column] = 10**4
    df.loc[(df[column] < 10**4) &(df[column] > 10**3),column] = 10**3
    df.loc[(df[column] < 10**3) &(df[column] > 10**2),column] = 10**2
    df.loc[(df[column] < 10**2),column] = 10**1
    df.loc[df[column].isna(),column] ='N/A'
    df[column]=df[column].astype(str)


#read ratings file, timestamp unused
ratings = pd.read_json(rating_file,lines=True).drop(columns='Timestamp')

#read item content file, drop unused columns
item_features = pd.read_json(content_file,lines=True).drop(columns=['DVD','BoxOffice','Website','Production','Response','Episode','Response','totalSeasons','seriesID','Ratings','Poster','Season','Rated','Released','Runtime'])

#generate nominations column (parse text on awards and transform to numeric)
item_features['Nominations']=item_features['Awards'].str.findall(r'[0-9]+ nomination').str.join(",").str.replace(r'[a-zA-Z]+','',regex=True)
item_features.loc[item_features['Nominations']=='','Nominations']=0
item_features['Nominations']=pd.to_numeric(item_features['Nominations'])

#generate wins column (parse text on awards and transform to numeric), specific award wins not considered
item_features['Wins']=item_features['Awards'].str.findall(r'[0-9]+ win').str.join(",").str.replace(r'[a-zA-Z]+','',regex=True)
item_features.loc[item_features['Wins']=='','Wins']=0
item_features['Wins']=pd.to_numeric(item_features['Wins'])

#Generate "original version" of some columns before turning them categorical
#these will be used for cold start users as well as to give weight to popularity later
item_features['imdbVotesOg'] = item_features['imdbVotes'].copy()
item_features['imdbRatingOg'] = item_features['imdbRating'].copy()
item_features['MetascoreOg'] = item_features['Metascore'].copy()
#convert votes to int
strToInt(item_features,'imdbVotesOg')
item_features['imdbVotesOg']=pd.to_numeric(item_features['imdbVotesOg'])


#Language features, filter languages that appear fewer than 200 times as other
item_features['Language'] = item_features['Language'].apply(lambda x: x.split(',')[0].replace(' ',''))
item_features.loc[item_features['Language']=='N/A', 'Language'] = "None"
item_features.loc[item_features['Language'].value_counts()[item_features['Language']].values < 200, 'Language'] = "Other"

#read predictions file
to_predict = pd.read_csv(target_file)

#prepare lightfm features
#genre
genre_features,unique_genres = oneHotEncoding(item_features,'Genre')
gen_feat=genFeatures(item_features,unique_genres)

#directors
director_features,_ = filterColumn(item_features,'Director','dir:')
dir_feat = genColumn(item_features,'Director','dir:')

#rating features
rating_features = strToInt(item_features,'imdbRating')
metascore_features = strToInt(item_features,'Metascore')

#awards 
wins_features,wins_feats = awardsFeats(item_features,'Wins','w:')

#language 
language_features = list(item_features['Language'].unique())

#votes
_=strToInt(item_features,'imdbVotes')
item_features['imdbVotes']=pd.to_numeric(item_features['imdbVotes'])
filterBig(item_features,'imdbVotes')
votes_features = strToInt(item_features,'imdbVotes')

#create features per item for lightfm
item_ids = item_features['ItemId'].to_list()
feat = [(item_ids[i],g) for i,g in enumerate(gen_feat)]
feat= [(feat[i][0],feat[i][1]+dfeat) for i,dfeat in enumerate(dir_feat)]
feat = [(feat[i][0],feat[i][1]+['imdbRating:'+j]) for i,j in enumerate(item_features['imdbRating'])]
feat = [(feat[i][0],feat[i][1]+['Metascore:'+j]) for i,j in enumerate(item_features['Metascore'])]
feat = [(feat[i][0],feat[i][1]+['imdbVotes:'+j]) for i,j in enumerate(item_features['imdbVotes'])]
feat = [(feat[i][0],feat[i][1]+[w]) for i,w in enumerate(wins_feats)]
feat = [(feat[i][0],feat[i][1]+[w]) for i,w in enumerate(item_features['Language'])]


#create list of possible features for lightfm
features = genre_features+director_features+rating_features+metascore_features+votes_features+wins_features+language_features

#build lightfm dataset
dataset = Dataset()
dataset.fit((x for x in ratings['UserId'].to_list()),
            (x for x in ratings['ItemId'].to_list()))
num_users, num_items = dataset.interactions_shape()
(interactions, weights) = dataset.build_interactions(((x['UserId'], x['ItemId'])
                                                    for x in ratings.to_dict(orient='records')))
dataset.fit_partial(items=(x for x in item_features['ItemId'].to_list()),
        item_features=features)
item_feat = dataset.build_item_features(feat)


#train lightfm
model = LightFM(loss='warp',random_state=3)
model.fit(interactions, item_features=item_feat)

#prepare predictions
user_id_map, user_feature_map, item_id_map, item_feature_map =dataset.mapping()


#predict coldstart value (sort by benchmarks)
def sort_by_features(df):
    predictions = pd.merge(df,item_features,on='ItemId')
    predictions.sort_values(by=['imdbVotesOg','Nominations','Wins','MetascoreOg','imdbRatingOg','ItemId'],ascending=[False,False,False,False,False,True],inplace=True)
    return predictions[['UserId','ItemId']]

#predict with lightfm
def sort_by_predictions(df):
    #convert users id to lightfm internal mapping
    users=df['UserId'].apply(lambda u:user_id_map[u]).values
    items=df['ItemId'].apply(lambda i:item_id_map[i]).values

    #predict scores
    predictions=model.predict(users,items,item_features=item_feat)
    df['Prediction']=predictions

    #standardize to allow for combination with item_features
    df['Prediction'] = (df['Prediction']-df['Prediction'].mean())/df['Prediction'].std()

    #give a weight to standardized item popularity in the dataset (as users in the database seem to favor popular items)
    df = pd.merge(df,item_features,on='ItemId')
    df['Prediction'] +=0.25*(df['imdbVotesOg']-df['imdbVotesOg'].mean())/df['imdbVotesOg'].std()

    #return user-item predictions sorted by rating, descending
    df.sort_values(by=['Prediction','ItemId'],ascending=[False,True],inplace=True)
    return df[['UserId','ItemId']]
    
def predict(to_predict):
    df_by_user={}
    #create a dataset for the predictions for each user
    for user, d in to_predict.groupby('UserId'):
        df_by_user[user] = d

    #predict for each user
    for user in df_by_user:
        #coldstart
        if not user in user_id_map:
            df_by_user[user]=sort_by_features(df_by_user[user])
        #not cold start
        else:
            df_by_user[user]=sort_by_predictions(df_by_user[user])
    #return user-item predictions in ascending userId and descending rating order
    return pd.concat(list(df_by_user.values()))
df=predict(to_predict)
#print results
#df.to_csv('aa.csv',index=False)
print('UserId,ItemId')
for prediction in df[['UserId','ItemId']].to_dict(orient='records'):
    print(prediction['UserId']+','+prediction['ItemId'])