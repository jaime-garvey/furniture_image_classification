import pandas as pd
import numpy as np
from pandas.io.json import json_normalize

from PIL import Image
import os
from io import BytesIO

import urllib3

urllib3.disable_warnings()
pd.set_option('display.max_colwidth', -1)


def read_data(path):
    '''
    Takes in path for data (json) and returns pandas dataframe
    '''
    return pd.read_json(path)


def json_to_tabular(data):
    '''
    Convert json columns to pandas dataframe
  
    Keyword Arguments:
    ------------------
    data - dataframe in json format
  
    Returns:
    --------
    a pandas dataframe
    '''
    images_df = pd.DataFrame.from_dict(json_normalize(data['images']), orient='columns')
    annotation_df = pd.DataFrame.from_dict(json_normalize(data['annotations']), orient='columns')
    combined_df = pd.merge(images_df, annotation_df, on='image_id')
    
    #take url out of list 
    combined_df['url'] = combined_df['url'].apply(lambda x: x[0])
    return combined_df


def save_jpg_from_url(image_row):
    '''
    Save images as jpg in directory
    
    Keyword Arguments:
    ------------------
    image_row - row of pandas dataframe
    '''
    
    image_url = image_row['url']
    image_id= image_row['image_id']

    #get file path
    train_directory_path = "drive/My Drive/Colab_Notebooks/computer_vision/iMaterialist_Challenge_Image_Classification/data/train_images"

    filename = 'image_' + str(image_id) + '.jpg'

    file_path = train_directory_path +'/' + filename

    #Save File to Folder
    try:
        http = urllib3.PoolManager(retries=False)
        result = http.request('GET',image_url)
        image_data = result.data

    except:
        print('Warning: Could not download image{}'.format(image_id))
        return

    try:
        open(file_path,"wb").write(image_data)
    except:
        print('Warning: Failed to save image {}'.format(image_id))
        return
    



