from time import time
from pymongo import MongoClient


class Mongo:
    def __init__(self, host, port, db, collection):
        self._host = host
        self._port = port
        self._db = db
        self._collection = collection
        
    def connect_init(self):
        self._client = MongoClient(self._host, self._port)
        self._collection = self._client[self._db][self._collection]

    def create_documents(self, document):
        self._collection.insert_one(document)
    
    def update_documents(self, timestamp, document, is_product_tf=False):
        if is_product_tf:
            self._collection.update({"timestamp" : timestamp}, {"$addToSet":{"product_tf" : document}})
        else:
            self._collection.update({"timestamp" : timestamp}, {"$addToSet":{"sections" : document}})

if __name__ == '__main__':
    mongo = Mongo('localhost', 27017, 'bm_elec', 'inspection1')
    mongo.connect_init()
    # documents = dict(
    #         timestamp = "ts",
    #         sections = "sec",
    #         product_tf = []
    #         )
    timestamp = "2021-12-07:15_28_39"
    document = [True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False]
    mongo.update_documents(timestamp=timestamp,document=document,is_product_tf=True)