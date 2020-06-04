# -*- coding: utf-8 -*-

from scrapy import Item, Field


class FeedScraperItem(Item):
    """
    A class used to create a storage container for data received by the feed scraper.
    """
    app_id = Field()
    app_name = Field()
    feed_url = Field()
    country = Field()
    review_id = Field()
    author = Field()
    author_url = Field()
    title = Field()
    rating = Field()
    content = Field()
    date = Field()
    app_version = Field()
    vote_count = Field()
    vote_sum = Field()