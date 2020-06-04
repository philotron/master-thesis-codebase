""" 
This module implements the AppfiguresBot class.
"""

import os
import json
import base64
import pandas as pd
import urllib.request


class AppfiguresBot():
    """
    A class used to create a software agent that connects to the Appfigures API
    in order to automatically scrape review data from the Appfigures database.
    """
    # information for accessing API
    username = 'appfigures_trail19@re-gister.com'
    password = 'mtdata19'
    client_key = "7a1ef7d33c564bb2ace51cea346299db"
    auth_string = bytes('%s:%s' % (username, password), 'ascii')
    base64_auth = base64.b64encode(auth_string).decode("ascii")
    api_url = "https://api.appfigures.com/v2/"
    dir_name = os.path.dirname(__file__)
    file_path = dir_name + "/../../data_files/app_mapping.csv"
    app_id_list = pd.read_csv(file_path).product_id

    def get_product_id(self, app_id):
        """
        Get the internal Appfigures product ID for a given App Store app ID.
        """
        path = "/products/apple/{}/?client_key={}".format(app_id, 
                                AppfiguresBot.client_key)
        request = urllib.request.Request(AppfiguresBot.api_url + path)
        request.add_header("Authorization", "Basic %s" % AppfiguresBot.base64_auth)
        result = json.load(urllib.request.urlopen(request))
        return (result["vendor_identifier"])
    
    def list_apps(self):
        """
        List all available apps that are tracked with the Appfigures account.
        """
        path = "products/mine?client_key={}".format(AppfiguresBot.client_key)
        request = urllib.request.Request(AppfiguresBot.api_url + path)
        request.add_header("Authorization", "Basic %s" % AppfiguresBot.base64_auth)
        result = json.load(urllib.request.urlopen(request))
        return (result)
    
    def send_request(self, app_id, country, count, start, end, page=1):
        """
        Request reviews of an app specified by id, country, max count and date.
        """
        path = "reviews?products={}&countries={}&count={}&page={}&start={}&end={}&client_key={}".format(app_id, country, count, page, start, end, 
        AppfiguresBot.client_key)
        request = urllib.request.Request(AppfiguresBot.api_url + path)
        request.add_header("Authorization", "Basic %s" % AppfiguresBot.base64_auth)
        result = json.load(urllib.request.urlopen(request))
        print(result["pages"])
        if result["this_page"] == result["pages"]:
            return result
        elif result["this_page"] < result["pages"]:
            result["reviews"] = result["reviews"] + (self.send_request(app_id, 
                  country, count, start, end, page + 1))["reviews"]
            return result
        else:
            return {"reviews": []}

    def fetch_reviews(self, country, count, start, end):
        """
        Fetch reviews for a list of app IDs and save results in CSV format. 
        """
        review_list = []
        for i in AppfiguresBot.app_id_list:
            print("Appending reviews for App ID:", i)
            result = self.send_request(i, country, count, start, end)
            review_list.extend(result["reviews"])           
        df = pd.DataFrame(review_list)  
        file_path = AppfiguresBot.dir_name + "/../../data_files/bot_data_1908_1910.csv"
        df.to_csv(file_path, encoding="utf-8", index=False)
        return df

