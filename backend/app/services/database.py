from pymongo import MongoClient
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class Database:
    client = None
    db = None

    @classmethod
    async def connect(cls):
        """
        Connect to MongoDB database
        """
        try:
            # Get MongoDB URI from environment variable
            mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
            
            # Connect to MongoDB
            cls.client = AsyncIOMotorClient(mongodb_uri)
            cls.db = cls.client.sign_language_db
            
            logger.info("Connected to MongoDB")
            
            # Create indexes
            await cls.create_indexes()
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            raise

    @classmethod
    async def close(cls):
        """
        Close MongoDB connection
        """
        if cls.client:
            cls.client.close()
            logger.info("Closed MongoDB connection")

    @classmethod
    async def create_indexes(cls):
        """
        Create necessary indexes
        """
        try:
            # Create indexes for users collection
            await cls.db.users.create_index("username", unique=True)
            await cls.db.users.create_index("email", unique=True)
            
            # Create indexes for translations collection
            await cls.db.translations.create_index([
                ("user_id", 1),
                ("created_at", -1)
            ])
            
            # Create indexes for sign_data collection
            await cls.db.sign_data.create_index([
                ("label", 1),
                ("created_at", -1)
            ])
            
            logger.info("Created MongoDB indexes")
            
        except Exception as e:
            logger.error(f"Failed to create indexes: {str(e)}")
            raise

    @classmethod
    async def get_database(cls):
        """
        Get database instance
        """
        if not cls.client:
            await cls.connect()
        return cls.db
