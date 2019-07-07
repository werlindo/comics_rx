# Libraries
import pandas as pd
import numpy as np

# Functions
def create_user_item_matrix(comic_ids, comic_factors):
    """
    Given list of user's comic preferences
    and a pandas df with item (comic) factors
    create a item matrix for the user
    """
    # Get rank
    num_latent_factors = len(comic_factors.features.iloc[0])
    
    # Initialize matrix
    comics_mtx = np.zeros(shape=(len(comic_ids), num_latent_factors)) 

    for index, comic in enumerate(comic_ids):
        #print(comic)
        #com_idx = comics_factors.loc[comics_factors['cxomic_id']==comic].index.values[0]
        comics_mtx[index, :] = np.array(comic_factors.loc[comic, 'features'])
    
    return comics_mtx

def create_user_impl_rate_matrix(comic_ids, ratings_list=None):
    """
    Given item matrix
    create implicit ratings matrix
    """
    if ratings_list is None:
        n = len(comic_ids)
        imp_rat_mtx = np.ones((n,1), 'int')
    else:
        imp_rat_mtx = np.array((ratings_list,)).T
    
    return imp_rat_mtx
    

def create_user_util_matrix(comics_matrix, user_rating_matrix):
    
    util_mtx = np.linalg.lstsq(comics_matrix, user_rating_matrix, rcond=None)

    # We just want the factors
    util_mtx = util_mtx[0].reshape((comics_matrix.shape[1],))

    return util_mtx

def make_n_comic_recommendations(comics, comic_factors, top_n):
    # Create item matrix
    comic_matrix = create_user_item_matrix(comic_ids=comics,
                                           comic_factors=comic_factors
                                          )
    
    # Create user matrix
    user_matrix = create_user_impl_rate_matrix(comic_ids=comics)
    
    # Create utility matrix
    utility_matrix = create_user_util_matrix(comic_matrix, user_matrix)
    
    # Update comic_factors dataframe for this user -> predicted scores!
    cf = comic_factors.copy()
    u = utility_matrix
    cf['pred'] = cf['features'].apply(lambda x: np.dot(x, u))
    
    # Get recommendations
    top_n_df = cf.sort_values(by=['pred'], ascending=False).head(top_n).copy()
    top_n_df.reset_index(inplace=True)
    top_n_df = top_n_df.loc[:,['comic_id','comic_title','img_url']].copy()
    
    return top_n_df