#!/usr/bin/python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import scrapy
from Crawler.items import LbcAppart
import unicodedata
import inspect


class AppartLBC_spider(scrapy.Spider):
    """

    """
    name = "appartLBC"
    allowed_domains = ["leboncoin.fr"]
    start_urls = [
        "http://www.leboncoin.fr/ventes_immobilieres/offres/ile_de_france/?f=a&th=1&ps=4&pe=10&sqs=3"
    ]

    def parse(self, response):
        curframe = inspect.currentframe()
        calframe = inspect.getouterframes(curframe, 2)
        print( 'caller name:\n')
        for frame in calframe:
            for i in frame:
                print( i )
            print( '' )

        return


        print("headers :")
        print(response.headers)
        print("")

        for list_lbc in response.xpath('//div[@class="list-lbc"]'):
            for html_a in list_lbc.xpath('a'):
                annonce = LbcAppart()
                annonce['htmlNode'] = html_a
                annonce['title']    = self.unicode_2_str( html_a.xpath('@title').extract()[0] )
                annonce['pageUrl']  = self.unicode_2_str( html_a.xpath('@href' ).extract()[0] )

                annonce['date'] = ''
                annonce['imageUrl'] = ''
                annonce['ville'] = ''
                annonce['price'] = ''
                self.parse_inner_node( html_a.xpath('./div/div'), annonce )

                yield annonce

    def parse_inner_node(self, div_dateDetails, annonce):
        for div_dateDetail in div_dateDetails:
            cssClass = div_dateDetail.xpath('@class')[0].extract()
            self.debug( 'cssClass : ' + str(cssClass) )

            if cssClass == 'date':
                date = ' '.join( div_dateDetail.xpath('./div/text()').extract() )
                annonce['date'] = self.unicode_2_str( date )

            elif cssClass == 'image':
                image = div_dateDetail.xpath('./div/img/@src')
                if len( image ) > 0:
                    annonce['imageUrl'] = self.unicode_2_str( image[0].extract() )

            elif cssClass == 'detail':
                self.parse_detail(div_dateDetail.xpath('./div'), annonce)

    def parse_detail(self, nodes, annonce):
        for div_villePrice in nodes:
            classVillePrice = div_villePrice.xpath('@class')[0].extract()
            self.debug( "Ville Price : " + str(classVillePrice) )

            if classVillePrice == 'placement':
                raw_string = div_villePrice.xpath('text()')[0].extract()
                raw_string = self.unicode_2_str( raw_string ).strip()
                ville = map(str.strip, raw_string.split('\n'))
                annonce['ville'] = ' '.join( ville )

            elif classVillePrice == 'price':
                price = filter(lambda x: x.isdigit(), div_villePrice.xpath('text()')[0].extract().strip() )
                annonce['price'] = self.unicode_2_str( price )

    def debug(self, text):
        return
        print("__DEBUG__ : " + str(text) )

    def unicode_2_str(self, text):
        return unicodedata.normalize('NFKD', text).encode('ascii','ignore')

