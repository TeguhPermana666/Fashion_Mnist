from sklearn.neighbors import NearestNeighbors


def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=6,algorithm='brute',metric='euclidean')
    neighbors.fit(feature_list)
    distances,indices = neighbors.kneighbors([features])
    
    return indices