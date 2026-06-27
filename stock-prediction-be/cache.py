"""
Redis cache helper for stock predictions.

Stores predictions as JSON strings in Redis with a TTL that expires
at midnight, so the cache auto-invalidates when new market data
becomes available the next trading day.
"""

import json
import os
import logging
from datetime import datetime, timedelta

import redis

logger = logging.getLogger(__name__)

REDIS_HOST = os.environ.get("REDIS_HOST", "redis")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))

_redis_client = None


def _get_redis() -> redis.Redis:
    """Lazy-initialize the Redis connection."""
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=0,
            decode_responses=True,
        )
    return _redis_client


def _cache_key(ticker: str) -> str:
    """Build a Redis key like 'prediction:AAPL'."""
    return f"prediction:{ticker.upper()}"


def _seconds_until_midnight() -> int:
    """Calculate seconds remaining until midnight (local time)."""
    now = datetime.now()
    midnight = (now + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return int((midnight - now).total_seconds())


def get_cached_prediction(ticker: str) -> dict | None:
    """
    Retrieve a cached prediction for the given ticker.

    Returns the prediction dict if found in Redis, or None if the
    cache is empty / expired.
    """
    try:
        r = _get_redis()
        key = _cache_key(ticker)
        data = r.get(key)
        if data is not None:
            logger.info(f"Cache HIT for {ticker}")
            return json.loads(data)
        logger.info(f"Cache MISS for {ticker}")
        return None
    except redis.ConnectionError:
        logger.warning("Redis unavailable — skipping cache lookup")
        return None


def set_cached_prediction(ticker: str, data: dict) -> None:
    """
    Store a prediction in Redis with a TTL that expires at midnight.

    This ensures that predictions are automatically invalidated when
    new market data becomes available the next day.
    """
    try:
        r = _get_redis()
        key = _cache_key(ticker)
        ttl = _seconds_until_midnight()
        r.setex(key, ttl, json.dumps(data))
        logger.info(f"Cached prediction for {ticker} (TTL: {ttl}s)")
    except redis.ConnectionError:
        logger.warning("Redis unavailable — skipping cache write")
