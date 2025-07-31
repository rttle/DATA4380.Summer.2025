#libraries
import pandas as pd
import numpy as np
import gzip
import json
import csv
from tqdm import tqdm

def is_watch_product(product):
    #'watch' in main_category
    main_cat=(product.get("main_category") or "").lower()
    if "watch" in main_cat:
        return True

    #'watch' in categories
    for cat_path in product.get("categories", []):
        if isinstance(cat_path, list):
            categories=cat_path
        else:
            categories=[cat_path]

        for category in categories:
            if "watch" in category.lower():
                return True

#filtering meta dataset
def meta_ds(file):
    watch_meta = {}

    with gzip.open(file, "rt", encoding="utf-8") as meta_file:
        for line in tqdm(meta_file, desc="Filtering metadata"):
            product=json.loads(line) #loading product meta data
            if is_watch_product(product):
                parent_asin=product.get("parent_asin")
                if parent_asin:
                    watch_meta[parent_asin]=product  
    return watch_meta

#filtering reviews dataset
def reviews_ds(file,watch_meta):
    reviews=[]
    watch_parent_asins=set(watch_meta.keys()) #create a set that has watch parent asins; set to help processing speed

    with gzip.open(file, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            review=json.loads(line)
            if review.get("parent_asin") in watch_parent_asins:
                reviews.append(review)
    return reviews

#full dataset
def full_ds(reviews,watch_meta):
    full_ds=[]

    for review in reviews:
        parent_asin=review.get("parent_asin")
        product_info=watch_meta.get(parent_asin)

        if product_info:
            #combine review columns + meta columns
            merged={**{f"review_{k}": v for k, v in review.items()},
                    **{f"meta_{k}": v for k, v in product_info.items()}}
            full_ds.append(merged)
    return full_ds

#create csv
def create_csv(full_ds):
    all_keys = set()
    for row in full_ds:
        all_keys.update(row.keys())

    fieldnames=sorted(all_keys)

    with open('amazon_watch_reviews.csv', "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(full_ds)

#preprocessing amazon reviews dataset to be readily used for creating synthetic labels (pros/cons list)
def llm_preprocessing(file1,file2):
    #filtered dataset
    watch_meta=meta_ds(file1)
    reviews=reviews_ds(file2,watch_meta)
    ds=full_ds(reviews,watch_meta)

    #csv
    create_csv(ds)

    #preprocessing
    df=pd.read_csv('amazon_watch_reviews.csv')
    reviews_grouped=df.groupby("meta_title")["review_text"].apply(list).reset_index()
    
    return reviews_grouped
