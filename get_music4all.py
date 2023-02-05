import numpy as np
import pandas as pd
import requests
import bz2
import gzip
import os
import sys
import re
import tarfile
import argparse
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# URL for the timestamp interactions dataset

def get_fname(url):

    url = url.split('?')[0]
    compressed_fname = url.split('/')[-1]
    split_name = compressed_fname.split('.')

    fname = '.'.join(split_name[:-1])
    ext = split_name[-1]

    return fname, ext

def download_files(url):

    fname, ext = get_fname(url)

    file_path = 'downloads/' + fname

    print(f'Filename: {file_path}')

    if os.path.exists(file_path):
        return 
    else:
        with open(file_path, "wb") as f:
            print("Downloading %s" % file_path)
            response = requests.get(url, stream=True)
            total_length = response.headers.get('content-length')

            if total_length is None: # no content length header
                f.write(response.content)
            else:
                if ext == 'bz2':
                    decomp = bz2.BZ2Decompressor()
                    dl = 0
                    total_length = int(total_length)
                    for data in response.iter_content(chunk_size=4096):
                        dl += len(data)

                        if decomp:
                            f.write(decomp.decompress(data))

                    done = int(50 * dl / total_length)
                    sys.stdout.write("\r|%s%s|" % ('■' * done, ' ' * (50-done)) )    
                    sys.stdout.flush()
                
                else:
                    f.write(gzip.decompress(response.content))

        if ext == 'gz':
            tar = tarfile.open(file_path, "r:")
            tar.extractall()
            tar.close()
        


def get_dataset_stats(data):

    user_actions = data.groupby(['user_id']).count()['track_id']

    num_users = len(data['user_id'].unique())
    num_tracks = len(data['track_id'].unique())

    unique_actions = data[['user_id', 'track_id']].drop_duplicates()

    unique_sparsity = 100 * (1 - round( (len(unique_actions)/(num_users * num_tracks)) , 4))

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

def read_music4all(chunksize=False, hidden_size=128):
    if chunksize:
        for chunk in pd.read_csv('userid_trackid_timestamp.tsv', sep='\t', chunksize=chunksize, parse_dates=True):
            #print(chunksize, len(chunk))
            get_dataset_stats(chunk)
            bert_emb, sentbert_emb = parse_lyrics(chunk, chunksize)

            return chunk, bert_emb, sentbert_emb
    else:
        data = pd.read_csv('userid_trackid_timestamp.tsv', sep='\t', parse_dates=True)
        get_dataset_stats(data)
        bert_emb, sentbert_emb = parse_lyrics(data, len(data))
        #audio_emb = get_audio(data)

        return data, bert_emb, sentbert_emb

def pca(id, data, hidden_size):

    X = StandardScaler().fit_transform(data)
    reducer = PCA(n_components=hidden_size)
    embedding = reducer.fit_transform(X)

    return pd.DataFrame({'track_id':id, 'embedding':embedding})


def get_audio(hidden_size):

    df = pd.read_csv('id_ivec1024.tsv', sep='\t')

    df = df.to_numpy()

    print(df.shape)

    ids = df[:,0]
    vectors = df[:,1:]

    #df = pca(ids, vectors, hidden_size)

    return df


def parse_lyrics(data, size):

    BERTtokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    BERT = BertModel.from_pretrained('bert-base-uncased') # 101 Start 102 End
    # For BERT CLS token is at the start

    SentBERT = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')

    lyric_dir = os.getcwd() + '/downloads/processed_lyrics/'

    track_ids = []
    embeddings = []
    emb_shapes = []
    sent_embeddings = []

    items = data['track_id'].astype(str) + '.txt'

    for name in items:
        id = name.split('.')[0]

        with open(lyric_dir + name, 'r') as f:

            lyrics = f.read()

            try:
                inputs = BERTtokenizer(lyrics, return_tensors="pt")
                outputs = BERT(**inputs)
                sent_emb = SentBERT.encode(lyrics)
            except:
                continue
            
            last_hidden_states = outputs.last_hidden_state.squeeze()

            shape = last_hidden_states.shape
            flattened_emb = last_hidden_states.detach().numpy().flatten()

            #print(type(flattened_emb), type(sent_emb))

            track_ids += id,
            emb_shapes += shape,
            embeddings += list(flattened_emb),
            sent_embeddings += list(sent_emb),

    if size < 10**6:  
        name = f'music4all_{size//10**3}k'
    else:
        name = f'music4all_{size//10**6}M'

    id_lyrics_bert = pd.DataFrame({'track_id':track_ids, 'shape':emb_shapes, 'embedding':embeddings})
    id_lyrics_sentbert = pd.DataFrame({'track_id':track_ids, 'embedding':sent_embeddings})

    id_lyrics_bert.to_pickle(f'{name}.bert_emb')
    id_lyrics_sentbert.to_pickle(f'{name}.sentbert_emb')

    #sent_embeddings_reduced = pca(track_ids, np.array(sent_embeddings), hidden_size)

    return id_lyrics_bert, id_lyrics_sentbert 
        
    #bart_df = pd.DataFrame.from_dict(id_bart_embedding_768)


def format_for_recbole(data, directory, rewrite=True):

    new_data = data.copy()
    new_data.timestamp = pd.to_datetime(data.timestamp).astype(int) / 10**9
    new_data.columns = ['user_id:token', 'track_id:token', 'timestamp:float']

    track_data = {'track_id:token':new_data['track_id:token'].unique()}
    unique_tracks = pd.DataFrame(track_data)
    user_data = {'user_id:token':new_data['user_id:token'].unique()}
    unique_users = pd.DataFrame(user_data)

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

    if rewrite:
        new_data.to_csv(f'{data_path}/{name}.inter' ,sep='\t', index=False)
        unique_users.to_csv(f'{data_path}/{name}.user' ,sep='\t', index=False)
        unique_tracks.to_csv(f'{data_path}/{name}.item' ,sep='\t', index=False)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--chunksize", "-ch", type=int, default=10**7, help="size of the dataset to be downloaded")
    
    args, _ = parser.parse_known_args()

    urls = [
        #'https://zenodo.org/record/6609677/files/userid_trackid_timestamp.tsv.bz2?download=1',
        'https://zenodo.org/record/6609677/files/processed_lyrics.tar.gz?download=1',
        'https://zenodo.org/record/6609677/files/id_ivec512.tsv.bz2?download=1'
        ]

    for url in urls:
        download_files(url)
    print('File Downloaded')
    print('Creating dataset for RecBole')
    data, bert_emb, sent_bert_emb = read_music4all(chunksize=args.chunksize)


    format_for_recbole(data, directory=os.getcwd()+'/dataset')

    print('Dataset Created\nDone :)')

