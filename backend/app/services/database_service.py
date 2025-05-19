from motor.motor_asyncio import AsyncIOMotorClient
from typing import Dict, List
import os

class DatabaseService:
    def __init__(self):
        # TODO: Get MongoDB URI from environment variables
        self.client = AsyncIOMotorClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
        self.db = self.client.sign_language
        self.signs_collection = self.db.signs
        
    async def get_sign_representation(self, text: str) -> Dict:
        """
        Get sign language representation for given text
        """
        sign = await self.signs_collection.find_one({"text": text.lower()})
        return sign if sign else None
    
    async def get_all_signs(self) -> List[Dict]:
        """
        Get all available signs from database
        """
        signs = await self.signs_collection.find().to_list(length=None)
        return signs
    
    async def add_sign(self, sign_data: Dict) -> Dict:
        """
        Add new sign to database
        """
        result = await self.signs_collection.insert_one(sign_data)
        return {"id": str(result.inserted_id)}

# Initialize database service
db_service = DatabaseService()
