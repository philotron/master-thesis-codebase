""" 
This module implements the AppleSpider class.
"""

import pandas as pd
from ..items import FeedScraperItem
from scrapy.spiders import Spider
from scrapy import Request
import os
import re
import time


class AppleSpider(Spider):
    """
    A class used to create a crawler that automatically scrapes Apple's RSS feed.
    """
    name = "spider"
    page_count = 10
    dir_name = os.path.dirname(__file__)
    file_path = dir_name + "/../../../data_files/app_mapping.csv"
    app_id_list = pd.read_csv(file_path).app_id 
    country_code_list = ["gb", "us"]
    domain = "https://itunes.apple.com/{}/rss/customerreviews/page={}/id={}/sortBy=mostrecent/xml"

    def start_requests(self):
        """
        Start multiple requests for a given list of parameters, namely app ID, country and page.
        """
        for a in AppleSpider.app_id_list:
            for b in AppleSpider.country_code_list:
                for c in range(1, AppleSpider.page_count+1):
                    print(a,b,c)
                    yield Request(url=AppleSpider.domain.format(b, str(c), a), callback=self.parse)

    def parse(self, response):
        """
        Parse XML data from RSS feed request and yield item object for each review.
        """
        time.sleep(0.5)
        item = FeedScraperItem()
        response.selector.remove_namespaces()
        entries = response.xpath("/feed/entry")

        for i in entries:
            # parse attributes
            feed_url = response.request.url
            app_id = re.search(r"id=(\d+)", feed_url).group()[3:]
            country = re.search(r"/[a-z]{2}/", feed_url).group()[1:-1]
            review_id = i.xpath("id/text()").extract_first()
            author = i.xpath("author/name/text()").extract_first()
            author_url = i.xpath("author/uri/text()").extract_first()
            title = i.xpath("title/text()").extract_first()
            rating = i.xpath("rating/text()").extract_first()
            content = i.xpath("content/text()").extract_first()
            date = i.xpath("updated/text()").extract_first()[:-6]
            app_version = i.xpath("version/text()").extract_first()
            vote_count = i.xpath("voteCount/text()").extract_first()
            vote_sum = i.xpath("voteSum/text()").extract_first()
            
            # store retrieved attribute values in container
            item["app_id"] = app_id
            item["review_id"] = review_id
            item["country"] = country
            item["feed_url"] = feed_url
            item["author"] = author
            item["author_url"] = author_url
            item["title"] = title
            item["rating"] = rating
            item["content"] = content
            item["date"] = date
            item["app_version"] = app_version
            item["vote_count"] = vote_count
            item["vote_sum"] = vote_sum
            yield item




