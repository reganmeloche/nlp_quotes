import requests
import random
from bs4 import BeautifulSoup
from pymongo import MongoClient
from datetime import datetime

from .Quote import Quote

class QuoteLib:
    def __init__(self, db):
        self.db = db
    
    def rate_quote(self, quote: Quote, rating: int, user_id: str) -> None:
        data = {
            'user_id': user_id,
            'quote': quote.__dict__,
            'rating': rating
        }
        self.db.insert_one(data)
        return
    
    def get_random_quote(self) -> Quote:
        random.seed(datetime.now())

        # Get pages of all authors
        r1 = requests.get('https://en.wikiquote.org/w/api.php?format=json&action=query&titles=List of people by name&generator=links&gplnamespace=0&gpllimit=20')
        json_data = r1.json()['query']['pages']
        keys = list(json_data.keys())
        random_key = random.choice(keys)

        # Select an author at random
        r2 = requests.get('https://en.wikiquote.org/w/api.php?action=query&format=json&origin=*&prop=links&pageids=' + random_key + '&redirects=1&pllimit=max')
        json_data2 = r2.json()['query']['pages']
        proper_id = list(json_data2.keys())[0]
        links = json_data2[proper_id]['links']
        random_author = random.choice(links)['title']
        while "List of people" in random_author:
            random_author = random.choice(links)['title']

        # From the random author, choose a random quote
        r3 = requests.get('https://en.wikiquote.org/w/api.php?action=parse&format=json&origin=*&prop=text&section=1&page=' + random_author)
        html_content = r3.text
        soup = BeautifulSoup(html_content, "lxml")
        quotes = []
        for t in soup.find_all('ul'):
            if t.parent.name != 'li':
                next_quote = t.find('li').text
                quotes.append(next_quote)
        
        the_quote = random.choice(quotes)
        return Quote('abc', the_quote, random_author)