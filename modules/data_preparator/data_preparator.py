""" 
This module implements the DataPreparator class.
"""

import nltk
nltk.download("stopwords")
import os
nltk.download('wordnet')
nltk.download('vader_lexicon')
import pandas as pd
from datetime import datetime as dati
from nltk.stem import WordNetLemmatizer 
import nltk.corpus as corpus


class DataPreparator():
    """
    A class used to implement all data preparation tasks.
    """
    feature_list = ["app_id", "app_version", "author", "title", 
                    "content", "country", "date", "rating"]
    special_chars =  '"!#$%&()*+,–-/:;<=>?@[\]^_`{|}~”“•.' 
    dir_name = os.path.dirname(__file__)
    file_path = dir_name + "/../../data_files/app_mapping.csv"
    app_mapping = pd.read_csv(file_path)

    def merge_datasets(self):
        """
        Harmonizes API bot and feed scraper datasets, returns a merged dataset 
        without duplicates and writes it to a CSV file.
        """
        # select subset of relevant features
        scraper_dataset = pd.read_csv(DataPreparator.dir_name + "/../../data_files/scraper_dataset.csv")
        scraper_dataset = scraper_dataset[DataPreparator.feature_list]
        # remove trailing zeros 
        scraper_dataset.app_version = scraper_dataset.app_version.str.rstrip(".00")
        # restructure the date format
        scraper_dataset.date = scraper_dataset.date.str.slice(0, -9) # remove time from date

        # rename certain features and then select subset of relevant features 
        bot_dataset = pd.read_csv(DataPreparator.dir_name + "/../../data_files/bot_dataset.csv")
        del bot_dataset["title"] # avoid duplicate columns
        bot_dataset = bot_dataset.rename(columns={"product_name": "app_name", 
                                                  "version": "app_version",
                                                  "original_title": "title", 
                                                  "original_review": "content", 
                                                  "iso": "country", 
                                                  "stars": "rating",
                                                  "vendor_id": "app_id"})
        bot_dataset = bot_dataset[DataPreparator.feature_list]
        # remove trailing zeros 
        bot_dataset.app_version = bot_dataset.app_version.str.rstrip(".00")
        # restructure the date format
        bot_dataset.date = bot_dataset.date.str.slice(0, -9)

        # concatenate the two datasets horizontally
        df = pd.concat([bot_dataset, scraper_dataset])
        
        # restrict records to predefined four-month time span
        df["date"] = pd.to_datetime(df["date"])  
        date_mask = (df['date'] >= dati.strptime("2019-08-19",'%Y-%m-%d')) \
        & (df['date'] <= dati.strptime("2019-12-19",'%Y-%m-%d'))
        df = df.loc[date_mask].reset_index(drop=True)
        
        # remove duplicate records
        duplicate_filter =  [["app_id", "app_version", "title", "content", "rating"],
                             ["app_id", "author", "title", "content", "rating"],
                             ["app_id", "author", "title", "content"]]
        # execute one filter after another to remove duplicates
        df = df.drop_duplicates(subset=duplicate_filter[0]).reset_index(drop=True) # handle date
        df = df.drop_duplicates(subset=duplicate_filter[1]).reset_index(drop=True) #  handle version
        df = df.drop_duplicates(subset=duplicate_filter[2], keep="first").reset_index(drop=True) # handle rating
        
        # join dataframe with app mapping table
        merged_df = pd.merge(left=df,right=DataPreparator.app_mapping, how="left", left_on="app_id", right_on="app_id")
        # delete redundant attributes
        del merged_df["product_id"] 
        del merged_df["url"]
        merged_df = merged_df.sort_index(axis=1)
        merged_df.to_csv(DataPreparator.dir_name + "/../../data_files/merged_dataset.csv", encoding="utf-8", index=False)
        return merged_df

    def preprocess_data(self):
        """
        Create document corpus and preprocess textual data for each document.
        """
        data = pd.read_csv(DataPreparator.dir_name + "/../../data_files/merged_dataset.csv")
        data["document"] = data["title"].map(str) + " " + data["content"].map(str)
        
        for col in ["document"]:
            # remove records that containing non-Latin characters 
            data = data[~data[col].str.contains("[\u0600-\u06FF]", regex=True)] # Arabic alphabet
            data = data[~data[col].str.contains("[\u4e00-\u9fff]", regex=True)] # Chinese alphabet
            print("Removing non-Latin reviews done.",dati.now().strftime('%Y-%m-%d %H:%M:%S'))
            
            # lowercase reviews
            data[col] = data[col].str.lower()
            print("Lowercasing reviews done.", dati.now().strftime('%Y-%m-%d %H:%M:%S'))
            
            # remove special characters
            data[col] = data[col].str.replace("’", "'")
            for char in DataPreparator.special_chars:
                data[col] = data[col].str.replace(char, " ") 
            print("Removing special chars done.", dati.now().strftime('%Y-%m-%d %H:%M:%S'))
            
            # remove stop words
            with open(DataPreparator.dir_name +"/custom_stop_word_list.txt", "r") as f:
                stop_words = f.read().splitlines()
            for word in stop_words:
                  data[col] = data[col].str.replace(r'\b'+word+r'\b', ' ', regex=True)
            print("Removing stop words done.", dati.now().strftime('%Y-%m-%d %H:%M:%S'))
            
            # apply lemmatization
            wordnet = corpus.wordnet
            wordnet._exception_map["n"]['ios'] = ['ios'] # exception for tagger
            lemmatizer = WordNetLemmatizer()
            data[col] = data[col].apply(lambda x: " ".join([lemmatizer.lemmatize(y) for y in x.split()]))
            print("Lemmatization done.", dati.now().strftime('%Y-%m-%d %H:%M:%S'))

            # remove whitespaces
            data[col] = data[col].str.replace("(?<=\d)\s(?=\d)", ".", regex=True)
            data[col] = data[col].str.replace("\s+", " ", regex=True)
            data[col] = data[col].str.strip()
            print("Removing whitespaces done.", dati.now().strftime('%Y-%m-%d %H:%M:%S'))
            
        data.to_csv(DataPreparator.dir_name + "/../../data_files/prepared_dataset.csv", encoding="utf-8", index=False)
        return data

    def assign_update_month(self):
        """
        Assign month number in relation to iOS 13 release date
        """
        def find_month_range(df):
            if df['date'] >= dati.strptime("2019-08-19",'%Y-%m-%d') and df['date'] <= dati.strptime("2019-09-18",'%Y-%m-%d'): return 0
            elif df['date'] >= dati.strptime("2019-09-19",'%Y-%m-%d') and df['date'] <= dati.strptime("2019-10-18",'%Y-%m-%d'): return 1
            elif df['date'] >= dati.strptime("2019-10-19",'%Y-%m-%d') and df['date'] <= dati.strptime("2019-11-18",'%Y-%m-%d'): return 2
            elif df['date'] >= dati.strptime("2019-11-19",'%Y-%m-%d') and df['date'] <= dati.strptime("2019-12-19",'%Y-%m-%d'): return 3
            else: return None
        
        df = pd.read_csv(DataPreparator.dir_name + "/../../data_files/prepared_dataset.csv")
        df["date"] = pd.to_datetime(df["date"])  
        df['update_month'] = df.apply(find_month_range, axis=1)
        df.to_csv(DataPreparator.dir_name + "/../../data_files/prepared_dataset.csv", encoding="utf-8", index=False)
        return df
        