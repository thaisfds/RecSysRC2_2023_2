import sys
import pandas as pd
from data import *
from model import *

def main():
    
    rating_path = sys.argv[1] if (len(sys.argv) >1) else "ratings.jsonl"
    content_path = sys.argv[2] if (len(sys.argv) >2) else "content.jsonl" 
    target_path = sys.argv[3] if (len(sys.argv) >3) else "targets.csv"

    content = pd.read_json(content_path, lines=True).drop(columns=['Rated', 'Released', 'Writer', 'Plot', 'Poster', 'Ratings','DVD', 'Production', 'Website', 'Response', 'totalSeasons', 'Season', 'Episode', 'Episode', 'seriesID', 'Type', 'Runtime'])
    ratings = pd.read_json(rating_path, lines=True).drop(columns=['Timestamp'])
    targets = pd.read_csv(target_path)

    features, data = preprocess_features(content)
    predictions = predict(content, ratings, targets, features, data)
    with open ('output.csv', 'w') as file:
      file.write('UserId,ItemId\n')
      for prediction in predictions[['UserId','ItemId']].to_dict(orient='records'):
          file.write(prediction['UserId']+','+prediction['ItemId']+'\n')

if __name__ == "__main__":
    main()