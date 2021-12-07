from model import Model
from bson.json_util import dumps

# MongoDB driver
import motor.motor_asyncio

client = motor.motor_asyncio.AsyncIOMotorClient('mongodb://localhost:27017/')
database = client.bm_elec
collection = database.inspection

async def fetch_last_one_result():
    result = []
    cursor = collection.find().sort([('timestamp', -1)]).limit(1)
    # list_cur = list(cursor)
    # json_data = dumps(list_cur, indent = 4)
    async for document in cursor:
        result.append(document)
    result = dumps(result)
    result = result[1:-1]
    return result

async def fetch_one_result(document):
    result = await collection.find_one(document, {'_id': 0})
    result = dumps(result)
    return result

async def fetch_all_results():
    result = []
    cursor = collection.find().sort([('timestamp', -1)]).limit(100)
    # async for document in cursor:
    #     results.append(Model(**document))
    async for document in cursor:
        result.append(document)
    # result = dumps(result)
    return result