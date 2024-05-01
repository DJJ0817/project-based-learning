import numpy as np 
from lightfm.datasets import fetch_movielens 
from lightfm import LightFM 
#from pprint import pprint

data = fetch_movielens(min_rating=4.0)


#print(type(data))

print(repr(data['train']))

model = LightFM(loss='warp')
#train model
model.fit(data['train'], epochs=30, num_threads=2)

def sample_recommendation(model, data, user_ids):

    n_users, n_itmes = data['train'].shape

    for user_id in user_ids: 
        
        #movie they already like
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]
        
        #movies predicts they like
        scores = model.predict(user_id, np.arange(n_itmes))
        print(scores)

        top_items = data['item_labels'][np.argsort(-scores)]

        print("User {}".format(user_id))
        print(" Known positives:")
        
        for x in known_positives[:4]:
            print(x)
        
        print("   Recommended:")

        for x in top_items[:3]:
            print("{}".format(x))

sample_recommendation(model, data, [3])


print(type(data['item_labels']))