""" Main module that controls the process flow of all developed modules.
"""

from scrapy.utils.project import get_project_settings
from scrapy.crawler import CrawlerProcess
from feed_scraper.spiders.apple_spider import AppleSpider
from appfigures_bot.appfigures_bot import AppfiguresBot
from data_preparator.data_preparator import DataPreparator
from data_processor.data_processor import DataProcessor


class Controller():
    """
    A class used to run each module in correct execution order.
    """

    def run_feed_scraper(self):
        """
        Starts a new scraping process for Apple's RSS feed.
        """
        process = CrawlerProcess(get_project_settings())
        process = CrawlerProcess(settings={
        "FEED_FORMAT": "csv",
        "FEED_URI": "scraper_data_new.csv"
        })
        process.crawl(AppleSpider)
        process.start()
    
    def run_appfigures_bot(self):
        """
        Starts a new scraping process for Appfigures' database.
        """
        process = AppfiguresBot()
        process.fetch_reviews("us,gb", "500", "2019-08-19", "2019-10-19")
    
    def run_data_preparation(self):
        """
        Starts the data preparation process.
        """
        process = DataPreparator()
        process.merge_datasets()
        process.preprocess_data()
        process.assign_update_month()   
    
    def run_data_processing(self):
        """
        Starts the data processing process.
        """
        process = DataProcessor()
        process.detect_sentiment()
        process.classify_reviews()   
        process.model_topics()   

    
if __name__ == "__main__":
    controller = Controller()
    
    #controller.run_feed_scraper()
    
    #controller.run_appfigures_bot()
    
    #controller.run_data_preparation()
    
    #controller.run_data_processing()
    
    
   


