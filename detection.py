import cv2
import pandas as pd
from matplotlib import pyplot as plt
import os
import numpy as np
from sklearn.model_selection import train_test_split

data_path = '.'
train_shelf_images = '../GroceryDataset/ShelfImages/train'
test_shelf_images = '../GroceryDataset/ShelfImages/test'
product_images = '../GroceryDataset/ProductImagesFromShelves/'


jpg_files = [f for f in os.listdir(f'{train_shelf_images}') if f.endswith('JPG')]
train_photos_df = pd.DataFrame(
    [[f, f[:6], f[7:14]] for f in jpg_files],
    columns=['file', 'shelf_id', 'planogram_id'])

jpg_files = [f for f in os.listdir(f'{test_shelf_images}') if f.endswith('JPG')]
test_photos_df = pd.DataFrame(
    [[f, f[:6], f[7:14]] for f in jpg_files],
    columns=['file', 'shelf_id', 'planogram_id'])

products_df = pd.DataFrame(
    [[f[:18], f[:6], f[7:14], i, *map(int, f[19:-4].split('_'))]
     for i in range(11)
     for f in os.listdir(f'{product_images}{i}') if f.endswith('png')],
    columns=['file', 'shelf_id', 'planogram_id','category', 'xmin', 'ymin', 'w', 'h'])

# convert from width height to xmax, ymax
products_df['xmax'] = products_df['xmin'] + products_df['w']
products_df['ymax'] = products_df['ymin'] + products_df['h']

shelves = list(set(train_photos_df['shelf_id'].values))
shelves_train, shelves_validation, _, _ = train_test_split(shelves, shelves, test_size=0.3, random_state=6)

def is_train(shelf_id): return shelf_id in shelves_train
train_photos_df['is_train'] = train_photos_df.shelf_id.apply(is_train)
products_df['is_train'] = products_df.shelf_id.apply(is_train)

df = products_df[products_df.category != 0].\
         groupby(['category', 'is_train'])['category'].\
         count().unstack('is_train').fillna(0)
df.plot(kind='barh', stacked=True)

train_photos_df.to_pickle(f'{data_path}photos.pkl')
products_df.to_pickle(f'{data_path}products.pkl')

def draw_shelf_photo(file):
    file_products_df = products_df[products_df.file == file]
    coordinates = file_products_df[['xmin', 'ymin', 'xmax', 'ymax']].values
    im = cv2.imread(f'{train_shelf_images}/{file}')
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    for xmin, ymin, xmax, ymax in coordinates:
        cv2.rectangle(im, (xmin, ymin), (xmax, ymax), (0, 255, 0), 5)
    cv2.imwrite("new.jpg", im)

# draw one photo to check our data
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
draw_shelf_photo('C3_P07_N1_S6_1.JPG')