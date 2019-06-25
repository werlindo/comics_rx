# %%
# This function helped to create the publisher short name dictionary (pub_dict).

# Get list of original publisher names
pub_list = list(trans_df['publisher'].unique())
pub_list

pub_dict = dict( (pub,'X') for pub in pub_list)
pub_dict

for key, val in pub_dict.items():
    new_val = input(key)
    pub_dict[key] = new_val

# Then just pasted the output of that to data_fcns.py
    
# %%


