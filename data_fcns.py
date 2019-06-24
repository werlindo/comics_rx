# Libraries
import pandas as pd

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