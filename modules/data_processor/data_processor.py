""" 
This module implements the DataProcessor class.
"""

import os
import nltk
nltk.download('wordnet')
nltk.download('vader_lexicon')
import pandas as pd
import pprint
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree.export import export_text
from sklearn.utils import shuffle
from scipy.stats import spearmanr
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import scipy.stats as stats
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import seaborn as sns
import random
from gensim import corpora, models


class DataProcessor():
    """
    A class used to implement all data mining and analysis tasks.
    """
    dir_name = os.path.dirname(__file__)
    file_path = dir_name + "/../../data_files/app_mapping.csv"
    app_mapping = pd.read_csv(file_path)

    def detect_sentiment(self):
        """
        Calculates sentiment scores for all documents and stores results in 
        newly created attribute column called sentiment_score.
        """
        df = pd.read_csv(DataProcessor.dir_name + "/../../data_files/prepared_dataset.csv")

        # initialize sentiment analyzer and apply it on documents
        sentiment_detector = SentimentIntensityAnalyzer()
        df["sentiment_score"] = None
        df["document"] = df["document"].astype(str)
        df["sentiment_score"] = df["document"].apply(lambda x: sentiment_detector.polarity_scores(str(x)))
        # store positive ratio from sentiment analysis
        df["sentiment_score"] = df["sentiment_score"].apply(lambda x: x.get("pos", None))
        
        # remove apps with less than 5 iOS 13 reviews
        df["flag"] = ((df.re_flag == 1) | (df.ml_flag == 1))    
        df_selection = df[df.flag == True]
        df_selection = df_selection.groupby(['app_name'])['document'].count().reset_index()
        df_selection = df_selection[df_selection.document>=5].reset_index()
        name_selection = set(df_selection.app_name)
        # group data frame by app and flag and calculate means of rating and sentiment columns
        df_grouped = df.groupby(['app_name', 'flag'])['rating', "sentiment_score"].mean().reset_index()
        df_grouped= df_grouped[df_grouped['app_name'].isin(name_selection)]
        
        # get continuous and dichotomous data columns
        rating = df_grouped.rating
        rating_means = (df_grouped[df_grouped.flag==False].rating.mean(),
                        df_grouped[df_grouped.flag==True].rating.mean())
        sentiment = df_grouped.sentiment_score
        sentiment_means = (df_grouped[df_grouped.flag==False].sentiment_score.mean(),
                        df_grouped[df_grouped.flag==True].sentiment_score.mean())
        flag = df_grouped.flag
        
        # calculate spearman correlation        
        print(spearmanr(flag, rating))
        print(spearmanr(flag, sentiment))
        
        # visualization of results
        # scatter plot for rating
        sns.set(rc={'figure.figsize':(5,6)})   
        sns.set_style('whitegrid', {'font.family':'serif', 'font.serif':'Times New Roman'})
        plot = sns.stripplot(x='flag', y='rating', data=df_grouped, orient="v", 
                             jitter=0.03, color="gray", size=6, alpha=0.8)
        plot.set(ylim=(1, 5))
        plt.plot([0, 1], [rating_means[0], rating_means[1]], linewidth=3, color="black",
                 linestyle="--")
        plot.set_xlabel("iOS 13 reference",fontsize=20)
        plot.set_ylabel("Average rating",fontsize=20)
        plot.tick_params(labelsize=15)
        
        # scatter plot for sentiment
        plt.clf()
        plt.cla()
        plt.close()
        sns.set(rc={'figure.figsize':(5,6)})   
        sns.set_style('whitegrid', {'font.family':'serif', 'font.serif':'Times New Roman'})
        plot = sns.stripplot(x='flag', y='sentiment_score', data=df_grouped, orient="v", 
                             jitter=0.03, color="gray", size=6, alpha=0.8)
        plot.set(ylim=(0, 0.6))
        plt.plot([0, 1], [sentiment_means[0], sentiment_means[1]], linewidth=3, color="black",
                 linestyle="--")
        plot.set_xlabel("iOS 13 reference",fontsize=20)
        plot.set_ylabel("Average sentiment score",fontsize=20)
        plot.tick_params(labelsize=15)
        
        df_grouped = df.groupby(['app_name'])['rating', "sentiment_score"].mean().reset_index()

        # scatter plot for rating and sentiment
        plt.clf()
        plt.cla()
        plt.close()
        sns.set(rc={'figure.figsize':(14,5)})   
        sns.set_style('whitegrid', {'font.family':'serif', 'font.serif':'Times New Roman'})
        plot = sns.regplot(x='rating', y='sentiment_score', data=df_grouped, color="black", 
                           scatter_kws={"color": "gray"}, line_kws={"color": "black"})
        plot.set(ylim=(0, 0.7), xlim=(1, 5))
        plot.set_xlabel("Average rating",fontsize=20)
        plot.set_ylabel("Average sentiment score",fontsize=20)
        plot.tick_params(labelsize=15)
        
        # save dataframe with annotated sentiment scores
        df.to_csv(DataProcessor.dir_name + "/../../data_files/processed_dataset.csv", encoding="utf-8", index=False)
   
    def save_load_model(self, model_name, save=0):
        """
        Saves and loads the trained random forest classification model.
        """
        if save:
            with open("rf_model" + '.pkl', 'wb') as f:
                pickle.dump(rf, f)    
        else:
            with open("rf_model" + '.pkl', 'rb') as f:
                rf = pickle.load(f)  

    def classify_reviews(self):
        """
        Classifies each review into either iOS 13-related or not iOS 13-related using 
        a naive REGEX approach and a more sophisticated machine learning approach.
        """
        df = pd.read_csv(DataProcessor.dir_name + "/../../data_files/processed_dataset.csv")
        df = df.fillna(value={'document': " "})
        df["re_flag"] = None # regular expression flag
        df["ml_flag"] = None # machine learning flag
        
        # regular expression approach for explicit iOS 13 references
        regex_string = r"((ios|iso|ois|osi|sio|soi|os)\s*13)|(13\s*(ios|iso|ois|osi|sio|soi|os))"
        df["re_flag"] = df.document.str.contains(regex_string,regex=True)
  
        # machine learning approach for implicit iOS 13 references
        # create training data set
        data = df
        sample_size = 21700
        data = pd.concat([data.sample(sample_size), 
                             data[data["re_flag"] == True]]).drop_duplicates().reset_index(drop=True)
        data = shuffle(data)
 
        # create tfidf vector as feature vector and another class label vector
        tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        x_tfidf_vector = tfidf_vectorizer.fit_transform(data["document"])
        y_vector = np.where(data['re_flag']==True, 1, 0)
        
        # create the same vectors for the entire dataset
        data = shuffle(df)
        data = data.fillna(value={'document': " "})
        x = tfidf_vectorizer.transform(data["document"])
        y = np.where(data['re_flag']==True, 1, 0)       
        
        # split data into train and test set
        x_train, x_test, y_train, y_test = train_test_split(x_tfidf_vector, y_vector, 
                                                    test_size=0.2, shuffle=False)
        
        # train multinomial naive Bayes model
        nb = MultinomialNB().fit(x_train, y_train)
        y_predicted = nb.predict(x_test)
        print("MultinomialNB")
        print(classification_report(y_test,y_predicted))
        print(pd.crosstab(y_test, y_predicted, rownames=['True'], colnames=['Predicted'], margins=True))
        print(accuracy_score(y_test, y_predicted))
        y_hats = nb.predict(x)
        print(classification_report(y,y_hats))
        print(pd.crosstab(y,y_hats, rownames=['True'], colnames=['Predicted'], margins=True))
        print(accuracy_score(y,y_hats))        
        
        # train logistic regression model
        lr = LogisticRegression(solver='lbfgs').fit(x_train, y_train)
        y_predicted = lr.predict(x_test)
        print("LogisticRegression")
        print(classification_report(y_test,y_predicted))
        print(pd.crosstab(y_test, y_predicted, rownames=['True'], colnames=['Predicted'], margins=True))
        print(accuracy_score(y_test, y_predicted))
        y_hats = lr.predict(x)
        print(classification_report(y,y_hats))
        print(pd.crosstab(y,y_hats, rownames=['True'], colnames=['Predicted'], margins=True))
        print(accuracy_score(y,y_hats))
        
        # train random forest model
        rf = RandomForestClassifier(n_estimators=50).fit(x_train, y_train)
        y_predicted = rf.predict(x_test)
        print("RandomForestClassifier")
        print(classification_report(y_test,y_predicted))
        print(pd.crosstab(y_test, y_predicted, rownames=['True'], colnames=['Predicted'], margins=True))
        print(accuracy_score(y_test, y_predicted))
        y_hats = rf.predict(x)
        print(classification_report(y,y_hats))
        print(pd.crosstab(y,y_hats, rownames=['True'], colnames=['Predicted'], margins=True))
        print(accuracy_score(y,y_hats))
        feature_names = tfidf_vectorizer.get_feature_names()
        # print randomly choosen decision tree of random forest model
        estimator = rf.estimators_[random.randrange(0, 50)]
        tree_rules = export_text(estimator, feature_names=feature_names)
        print(tree_rules)
 
        # conduct five-fold cross validation
        models = [
            RandomForestClassifier(n_estimators=50),
            MultinomialNB(),
            LogisticRegression(),
        ]
        cv_fold = 5
        cv_df = pd.DataFrame(index=range(cv_fold * len(models)))
        entries = []
        for model in models:
          model_name = model.__class__.__name__
          accuracies = cross_val_score(model, x_tfidf_vector, y_vector, 
                                       scoring='accuracy', cv=cv_fold)
          for fold_idx, accuracy in enumerate(accuracies):
              entries.append((model_name, fold_idx, accuracy))
        cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
        
        # visualization of results
        # print boxplots with model accuracies
        sns.set(rc={'figure.figsize':(14,6)})   
        sns.set_style('whitegrid', {'font.family':'serif', 'font.serif':'Times New Roman'})
        plot = sns.boxplot(x='accuracy', y='model_name', color="0.90", data=cv_df, 
                    order=["MultinomialNB","LogisticRegression","RandomForestClassifier"],
              orient="h", linewidth=3)
        
        sns.swarmplot(x='accuracy', y='model_name', data=cv_df, 
              size=10, edgecolor="gray", color="black", 
              linewidth=1, order=["MultinomialNB", "LogisticRegression",
                                  "RandomForestClassifier"],
                                  orient="h")
        plot.set_xlabel("Accuracy",fontsize=25)
        plot.set_ylabel("Model name",fontsize=25)
        plot.tick_params(labelsize=20)
           
        # store predictions of the two classification approaches       
        data.loc[:,"ml_flag"] = y_hats # machine learning predictions
        data["re_flag"] = data["re_flag"].astype(int) # regex predictions
        data = data.reset_index(drop=True)
        
        data.to_csv(DataProcessor.dir_name + "/../../data_files/processed_dataset.csv", encoding="utf-8", index=False)
        return df
    
    def model_topics(self):
        """
        Carries out topic modeling by using latent Dirichlet allocation algorithm.
        """
        df = pd.read_csv(DataProcessor.dir_name + "/../../data_files/processed_dataset.csv")
        # create subset of all classified ios 13 reviews
        data = df[(df.re_flag == 1) | (df.ml_flag == 1)].reset_index(drop=True)        
        # greate ungrouped list of documents
        ungrouped_docs = data.document.to_list()
        # tokenize list of documents into list of words for each document
        token_docs = [(sen).split() for sen in ungrouped_docs]
        
        try:   
            # try to load id2word dictionary and LDA model
            with open(DataProcessor.dir_name + "/../../data_files/id2word_dictionary", "rb") as f:
                id2word_dict = pickle.load(f)
            lda_model =  models.LdaModel.load(DataProcessor.dir_name + '/../../data_files/lda_model3') 
            # build bag of words corpus from documents and id2word dictionary        
            bow_corpus = [id2word_dict.doc2bow(text) for text in token_docs]       
        except:    
            # build bigrams of words occuring at least min_count times
            bigram = models.Phrases(token_docs, min_count=100)
            bigram_mod = models.phrases.Phraser(bigram)
            # apply bigram model on data words
            token_docs  = [bigram_mod[doc] for doc in token_docs] 
            # build dictionary with an id for each word 
            id2word_dict = corpora.Dictionary(token_docs)
            # filter out extreme values from the dictionary
            id2word_dict.filter_extremes(no_below=2, no_above=0.5)
            # save newly created id2word dictionary
            with open("id2word_dictionary", "wb") as f:
                pickle.dump(id2word_dict, f)

            # build bag of words corpus from documents and id2word dictionary        
            bow_corpus = [id2word_dict.doc2bow(text) for text in token_docs]
                   
            # train LDA model with 6 topics and 50 training iterations
            lda_model = models.ldamodel.LdaModel(corpus=bow_corpus,
                                                       id2word=id2word_dict,
                                                       num_topics=6,
                                                       passes=50,
                                                       alpha='auto')
            # save created LDA model
            lda_model =  models.LdaModel.load('lda_model')
            # print topic results
            pp = pprint.PrettyPrinter(indent=3)
            pp.pprint(lda_model.print_topics())
            # compute perplexity
            print('\nPerplexity: ', lda_model.log_perplexity(bow_corpus))

            coherence_model_lda = models.CoherenceModel(model=lda_model, 
                                                        corpus=bow_corpus, 
                                                        dictionary=id2word_dict, 
                                                        coherence='c_v',
                                                        texts=token_docs)
            coherence_lda = coherence_model_lda.get_coherence()
            print('\nCoherence Score: ', coherence_lda)
        
        # assign the most probable topic to each document
        data["lda_topic"] = None
        for index, row in data.iterrows():
            data.loc[index,"lda_topic"] = sorted(lda_model[bow_corpus[index]], 
                                    reverse=True, key=lambda x: x[1])[0][0]+1
        # assign the overarching issue type to each document
        data.loc[(data["lda_topic"] == 1) | (data["lda_topic"] == 6), "issue_type"] = "feature request" 
        data.loc[(data["lda_topic"] == 2) | (data["lda_topic"] == 3) | (data["lda_topic"] == 5), "issue_type"] = "functional error"
        data.loc[(data["lda_topic"] == 4), "issue_type"] = "device compatibility"
        
        # visualize the topics with LDAvis framework
        pyLDAvis.enable_notebook()
        vis = pyLDAvis.gensim.prepare(lda_model, bow_corpus, id2word_dict)
        pyLDAvis.show(vis)

        # visualize rating distribution over topics with boxplots
        sns.set(rc={'figure.figsize':(16,4)})   
        #sns.set_style("white")
        sns.set_style('whitegrid', {'font.family':'serif', 'font.serif':'Times New Roman'})
        #plt.figure(figsize=(12,6))
        plot = sns.boxplot(x='issue_type', y='rating', color="0.90", data=data, linewidth=3,
                           medianprops={'color':'black', 'linewidth':'4'})
        plot.set_xlabel("Issue type",fontsize=20)
        plot.set_ylabel("Rating",fontsize=20)
        plot.tick_params(labelsize=15)

        # visualize positivity score distribution over topics with boxplots
        sns.set(rc={'figure.figsize':(14,6)})   
        #sns.set_style("white")
        sns.set_style('whitegrid', {'font.family':'serif', 'font.serif':'Times New Roman'})
        #plt.figure(figsize=(12,6))
        plot = sns.boxplot(x='issue_type', y='sentiment_score', color="0.90", data=data, linewidth=3,
                           medianprops={'color':'black', 'linewidth':'4'})
        plot.set_xlabel("Issue type",fontsize=25)
        plot.set_ylabel("Sentiment score",fontsize=25)
        plot.tick_params(labelsize=20)
    
        # store dataframe with assigned topics    
        data.to_csv(DataProcessor.dir_name + "/../../data_files/lda_dataset.csv", encoding="utf-8", index=False)    
        df = pd.read_csv(DataProcessor.dir_name + "/../../data_files/lda_dataset.csv") #preprocessed dataset
        
        # group documents by issue type
        feature_request = df[df.issue_type=="feature request"]        
        functional_error = df[df.issue_type=="functional error"]        
        device_compatibility = df[df.issue_type=="device compatibility"]

        # identify all authors that appear across at least two issue groups
        author_filter = list(set(feature_request.author) & (set(functional_error.author))) \
            + list(set(feature_request.author) & (set(device_compatibility.author))) \
            + list(set(functional_error.author) & (set(device_compatibility.author)))
        # filter out reviews from corresponding authors
        feature_request = feature_request[~feature_request['author'].isin(author_filter)]
        functional_error = functional_error[~functional_error['author'].isin(author_filter)]
        device_compatibility = device_compatibility[~device_compatibility['author'].isin(author_filter)]

        # perform Kruskal-Wallis test statistics for rating
        print(stats.kruskal(feature_request.rating, 
                            functional_error.rating, 
                            device_compatibility.rating))
        print(stats.kruskal(functional_error.rating, 
                            device_compatibility.rating))
        
        # perform Kruskal-Wallis test statistics for sentiment score
        print(stats.kruskal(feature_request.sentiment_score, 
                            functional_error.sentiment_score, 
                            device_compatibility.sentiment_score))
        print(stats.kruskal(functional_error.sentiment_score, 
                            device_compatibility.sentiment_score))
        return data
    
    
    
 

