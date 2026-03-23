from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv

from app.models import User, Player

load_dotenv()

client: AsyncIOMotorClient = None
database = None

async def init_database():
    """Initialize Beanie with MongoDB. Gracefully skips if MongoDB is unavailable."""
    global client, database

    mongodb_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    database_name = os.getenv("DATABASE_NAME", "trade_analyzer")

    try:
        # Create motor client with short timeout
        client = AsyncIOMotorClient(mongodb_url, serverSelectionTimeoutMS=3000)
        database = client[database_name]

        # Initialize Beanie
        await init_beanie(
            database=database,
            document_models=[User, Player]
        )
        print("MongoDB connected successfully")
    except Exception as e:
        print(f"MongoDB not available ({e}). Predictions API will still work.")
        client = None
        database = None

    return database

async def close_database():
    """Close database connection"""
    global client
    if client:
        client.close()

async def get_database():
    """Get database instance"""
    global database
    if database is None:
        await init_database()
    return database

