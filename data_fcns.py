#-------------------------------------
# Libraries
#------------------------------------
import pandas as pd

#-------------------------------------
# Functions
#------------------------------------
# Original source and credit:
# https://stackoverflow.com/questions/29882573/pandas-slow-date-conversion
def date_converter(s):
    """
    This is an extremely fast approach to datetime parsing.
    For large data, the same dates are often repeated. Rather than
    re-parse these, we store all unique dates, parse them, and
    use a lookup to convert all dates.
    """
    dates = {date:pd.to_datetime(date) for date in s.unique()}
    return s.map(dates)

def cut_issue_num(title_and_num):
    """Remove #-sign and all characters to the right.
    
    Parameters:
    -----------
    title_and_num = string representing title, issue num and any other details.
    """
    return title_and_num[:title_and_num.rfind('#')].strip()


#-------------------------------------
# References
#------------------------------------
# Dictionary to map to shorter publisher names.
pub_dict = {'Amaze Ink Slave Labor Graphics': 'SLG',
             'Archie Comics': 'Archie',
             'Aspen MLT': 'Aspen',
             'Avatar Press': 'Avatar',
             'Bongo Comics': 'Bongo',
             'Boom! Studios': 'Boom',
             'D.D.P.': 'DDP',
             'D.E.': 'DE',
             'Dark Horse': 'Dark Horse',
             'DC Comics': 'DC',
             'DC Vertigo': 'Vertigo',
             'DC Wildstorm': 'Wildstorm',
             'Drawn & Quarterly': 'D&Q',
             'Fantagraphics': 'Fantagraphics',
             'IDW Publishing': 'IDW',
             'Image Comics': 'Image',
             'Image Topcow': 'Topcow',
             'Marvel Comics': 'Marvel',
             'Oni Press': 'Oni',
             'Other': 'Other',
             'Radical Publishing': 'Radical',
             'Red 5 Comics': 'Red 5',
             'Top Shelf Productions': 'Top Shelf',
             'Zenescope Entertainment': 'Zenescope'
           }