import numpy as np
import pandas as pd
import requests
import bz2
import os
import sys
import re

# URL for the timestamp interactions dataset

def get_fname(url):

    url = url.split('?')[0]
    compressed_fname = url.split('/')[-1]
    fname = '.'.join(compressed_fname.split('.')[:-1])

    return fname

def download_file(url):

    file_name = get_fname(url)

    with open(file_name, "wb") as f:
        print("Downloading %s" % file_name)
        response = requests.get(url, stream=True)
        total_length = response.headers.get('content-length')

        if total_length is None: # no content length header
            f.write(response.content)
        else:
            dl = 0
            total_length = int(total_length)
            for data in response.iter_content(chunk_size=4096):
                dl += len(data)
                f.write(data)
                done = int(50 * dl / total_length)
                sys.stdout.write("\r|%s%s|" % ('■' * done, ' ' * (50-done)) )    
                sys.stdout.flush()

def get_dataset_stats(data):

    user_actions = data.groupby(['user_id']).count()['track_id']
    item_plays = data.groupby(['track_id']).count()['user_id']

    num_users = len(data['user_id'].unique())
    num_tracks = len(data['track_id'].unique())

    unique_actions = data[['user_id', 'track_id']].drop_duplicates()

    unique_sparsity = 100 * (1 - round( (len(unique_actions)/(num_users * num_tracks)) , 4))
    sparsity = 100 * (1 - round( (len(data)/(num_users * num_tracks)) , 4))

    #------------------------------------------------------------------------------
    """
    print(f'''
    Interactions: {len(data)}
    Unique Interactions: {len(unique_actions)}
    Num Users: {num_users}
    Num Tracks: {num_tracks}
    Earliest Interaction: {max(data.timestamp)}
    Latest Interaction: {min(data.timestamp)}
    
    There are {len(user_actions)} users.
    There are {sum(user_actions<listen_cap)} users with less than {listen_cap} listens
    Each user has an average of {int(np.mean(user_actions))} listens.

    There are {len(item_plays)} tracks.
    Each track has an average of {int(np.mean(item_plays))} plays.
    Unique Interaction Data Sparsity: {unique_sparsity}%
    Sparsity with Repeated Interactions: {sparsity}%

    ''')
    """

    #------------------------------------------------------------------------------
    # MAIN STATISTICS

    size = len(data)

    if size < 10**6:
        print(f'\nMusic4All {len(data)//10**3}K Interactions Stats')
    else:
        print(f'\nMusic4All {len(data)//10**6}M Interactions Stats')
        

    print(f'''
    |Users\t| Items\t\t| Interactions\t| Average Length\t| Median Length\t| Sparsity\t|
    -----------------------------------------------------------------------------------------------------
    |{num_users}  \t| {num_tracks}\t\t| {len(data)}\t| {np.mean(user_actions).astype(int)}\t\t\t| {np.median(user_actions).astype(int)}\t\t| {round(unique_sparsity,2)}\t\t|
    
    ''')

def read_music4all(chunksize=False):
    if chunksize:
        for chunk in pd.read_csv('userid_trackid_timestamp.tsv', sep='\t', chunksize=chunksize, parse_dates=True):
            get_dataset_stats(chunk)
            return chunk
    else:
        data = pd.read_csv('userid_trackid_timestamp.tsv', sep='\t', parse_dates=True)
        get_dataset_stats(data)
        return data


def format_for_recbole(data, directory):

    new_data = data.copy()
    new_data.timestamp = pd.to_datetime(data.timestamp).astype(int) / 10**9
    new_data.columns = ['user_id:token', 'track_id:token', 'timestamp:float']

    size = len(new_data)
    if size < 10**6:  
        name = f'music4all_{size//10**3}k'
    else:
        name = f'music4all_{size//10**6}M'

    data_path = f'{directory}/{name}'

    if os.path.exists(data_path):
        pass
    else:
        os.mkdir(data_path)

    new_data.to_csv(f'{data_path}/{name}.inter' ,sep='\t', index=False)
    new_data['user_id:token'].to_csv(f'{data_path}/{name}.user' ,sep='\t', index=False)
    new_data['track_id:token'].to_csv(f'{data_path}/{name}.item' ,sep='\t', index=False)

if __name__ == '__main__':

    print('Downloading File...')
    interactions_url = 'https://zenodo.org/record/6609677/files/userid_trackid_timestamp.tsv.bz2?download=1'
    url = 'https://zenodo.org/record/6609677/files/processed_lyrics.tar.gz?download=1'

    download_file(url)
    print('File Downloaded')
    print('Creating dataset for RecBole')
    data = read_music4all(chunksize=10**6)
    format_for_recbole(data, directory=os.getcwd()+'/dataset')
    print('Dataset Created\nDone :)')

