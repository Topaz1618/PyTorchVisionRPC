import redis
from config import REDIS_HOST


def redis_conn():
    r = redis.Redis(host=REDIS_HOST, port=6379)
    return r
