"""
Create DynamoDB table for semantic cache.
Run once: python scripts/create_dynamodb_table.py
"""

import boto3
import os
from dotenv import load_dotenv

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
TABLE_NAME = os.getenv("CACHE_TABLE", "rag-semantic-cache")

def create_table():
    dynamodb = boto3.client("dynamodb", region_name=AWS_REGION)
    
    try:
        # Create table
        dynamodb.create_table(
            TableName=TABLE_NAME,
            KeySchema=[
                {"AttributeName": "cache_key", "KeyType": "HASH"}
            ],
            AttributeDefinitions=[
                {"AttributeName": "cache_key", "AttributeType": "S"}
            ],
            BillingMode="PAY_PER_REQUEST"  # No capacity planning needed
        )
        print(f"✓ Creating table '{TABLE_NAME}'...")
        
        # Wait for table to be active
        waiter = dynamodb.get_waiter("table_exists")
        waiter.wait(TableName=TABLE_NAME)
        print(f"✓ Table '{TABLE_NAME}' is now active")
        
        # Enable TTL
        dynamodb.update_time_to_live(
            TableName=TABLE_NAME,
            TimeToLiveSpecification={
                "Enabled": True,
                "AttributeName": "ttl"
            }
        )
        print(f"✓ TTL enabled on 'ttl' attribute (items auto-expire)")
        
        print(f"\n✅ DynamoDB table ready: {TABLE_NAME}")
        
    except dynamodb.exceptions.ResourceInUseException:
        print(f"⚠ Table '{TABLE_NAME}' already exists")
    except Exception as e:
        print(f"✗ Error: {e}")


if __name__ == "__main__":
    create_table()
