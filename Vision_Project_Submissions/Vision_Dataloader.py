#libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import data as tf_data
from keras import layers
from PIL import Image
import io

def convert_image(bytes):
    image=Image.open(io.BytesIO(bytes)).convert('RGB')
    return np.array(image)

def image_preprocess(dataframe):
    #changing column type (object to datetime)
    dataframe['text']=pd.to_datetime(dataframe['text'],format='%H:%M')

    #splitting into two columns
    dataframe['Hour']=dataframe['text'].dt.hour+dataframe['text'].dt.minute / 60.0 #decimal will hopefully help give more information to model
    dataframe['Hour (categorical)']=dataframe['text'].dt.hour #hour column for categorical approach
    dataframe['Minute']=dataframe['text'].dt.minute

    #target column option 2: minutes from midnight
    dataframe['Minutes from Midnight']=(dataframe['Hour (categorical)']*60)+dataframe['Minute']

    #Changing dictionary to just keep values
    dataframe['image']=dataframe['image'].apply(lambda x: x['bytes'])

    #convert byte to image
    dataframe['image']=dataframe['image'].apply(convert_image)

    #convert to tensor
    dataframe['image']=dataframe['image'].apply(lambda x: tf.convert_to_tensor(x, dtype=tf.float32)/255) #conversion + normalize

    return dataframe

#data augmentation setup
data_augmentation_layers = [
    layers.RandomContrast(0.2),
    layers.RandomRotation(0.1),
    layers.RandomTranslation(0.15,0.15)
]

def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images

def image_loader(file_name,target,batches):
    df=pd.read_parquet(file_name)
    df=image_preprocess(df)

    #labels
    images=list(df['image'])
    if target=='minutes':
        labels=df['Minutes from Midnight'].astype('float32').values
    elif target=='list':
        labels=df[['Hour','Minute']].astype('float32').values
    elif target=='categorical':
        labels=df[['Hour (categorical)','Minute']]

    #changing pandas dataframe to tf dataset; taking subset of dataset
    df100=df[:100]
    images=list(df100['image'])
    labels=labels[:100].astype('float32').values
    tf_dataset=tf.data.Dataset.from_tensor_slices((images,labels)).batch(64)

    #taking small subset for model + splitting into training and validation
    train_ds=tf_dataset.take(80)
    val_ds=tf_dataset.skip(80)

    train_ds = tf_dataset.map(
        lambda img, label: (data_augmentation(img), label),
        num_parallel_calls=tf_data.AUTOTUNE,
    )

    train_ds=train_ds.batch(batches)
    val_ds=val_ds.batch(batches)

    return train_ds, val_ds
