#!/usr/bin/python2
# -*- coding: utf-8 -*-

from __future__ import print_function
import scrapy
from scrapy.crawler import CrawlerProcess
from Crawler.spiders import AppartLBC_spider


process = CrawlerProcess({
    'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)'
})

process.crawl(AppartLBC_spider)
process.start() # the script will block here until the crawling is finished

print( 'END' )


