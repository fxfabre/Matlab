# -*- coding: utf-8 -*-


import scrapy

class LbcAppart(scrapy.Item):
    htmlNode = scrapy.Field()
    title = scrapy.Field()
    pageUrl = scrapy.Field()

    ville = scrapy.Field()
    price = scrapy.Field()
    date = scrapy.Field()
    imageUrl = scrapy.Field()


