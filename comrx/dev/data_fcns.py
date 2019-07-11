# -------------------------------------
# Libraries
# ------------------------------------
import pandas as pd
import boto3

# -------------------------------------
# Functions
# ------------------------------------
# Original source and credit:
# https://stackoverflow.com/questions/29882573/pandas-slow-date-conversion


def date_converter(s):
    """
    This is an extremely fast approach to datetime parsing.
    For large data, the same dates are often repeated. Rather than
    re-parse these, we store all unique dates, parse them, and
    use a lookup to convert all dates.
    """
    dates = {date: pd.to_datetime(date) for date in s.unique()}
    return s.map(dates)


def cut_issue_num(title_and_num):
    """
    Remove #-sign and all characters to the right.
    Parameters:
    -----------
    title_and_num = string representing title, issue num and any other details.
    """
    return title_and_num[:title_and_num.rfind('#')].strip()


def make_int(curr_str):
    """Take a string and make it an int. No error trapping at this time."""
    return int(curr_str)


def update_urls(tgt_titles, client, work, cur, conn):
    """Update URL for comic in comic table in PSQL RDS, given
       list of comic titles
    """
    generic_img_url = ('https://comrx.s3-us-west-2.amazonaws.com' +
                       '/covers/_no_cover_.jpg')
    for title in tgt_titles:
        # Find file name

        # escape single count
        title_sql = title.replace("'", "''")

        try:
            got_filename = work.loc[work['comic_title']
                                    == title]['filename'].values[0]
            client.head_object(Bucket='comrx', Key='covers/' + got_filename)
            print('got ' + got_filename)

            # Update comics table on AWS
            img_url = work.loc[work['comic_title']
                               == title]['search_path'].values[0]
            print(img_url)

            # Build query string
            query_update = ("UPDATE comics " +
                            "SET img_url = '" + img_url + "' " +
                            "WHERE comic_title = '" + title_sql +
                            "';"
                            )
            print(query_update)

            # Execute query
            cur.execute(query_update)
            conn.commit()

        except IndexError:
            print('No Match, use generic image')
            # Build query string
            query_update = ("UPDATE comics " +
                            "SET img_url = '" + generic_img_url + "' " +
                            "WHERE comic_title = '" + title_sql +
                            "';"
                            )
            print(query_update)

            # Execute query
            cur.execute(query_update)
            conn.commit()


def update_manual_img_url(comic_title, new_url, conn):
    """
    Given comic title, update the url to the image.
    Uses psycopg2 connection
    """
    # escape single count
    title_sql = comic_title.replace("'", "''")

    # Open cursor
    cur = conn.cursor()

    # Form query
    query = ("UPDATE comics " +
             "SET img_url = '" + new_url + "' " +
             "WHERE comic_title = '" + title_sql +
             "';")
    # Execute
    cur.execute(query)
    conn.commit()

    # Check
    query = ("SELECT * FROM comics " +
             "WHERE comic_title = '" + title_sql +
             "';")
    # Execute
    cur.execute(query)
    conn.commit()

    # Check results
    temp_df = pd.DataFrame(cur.fetchall())
    temp_df.columns = [col.name for col in cur.description]

    return temp_df

# -------------------------------------
# References
# ------------------------------------


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
