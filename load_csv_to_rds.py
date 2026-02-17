"""Simple script to load CSV into AWS RDS MySQL."""

import pandas as pd
import pymysql
from sqlalchemy import create_engine

# Configuration
DB_HOST = "ticket-database.c27w604qw22b.us-east-1.rds.amazonaws.com"
DB_USER = "jemima"
DB_PASSWORD = "jemima1998"
DB_NAME = "ticket_database"

# CSV file
CSV_FILE = "Data/Raw_Ticket_Jan01_2025_TO_Jan20_2026.csv"

# Create database if not exists
conn = pymysql.connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD)
conn.cursor().execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
conn.close()
print(f"Database '{DB_NAME}' ready.")

# Connect and load
engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:3306/{DB_NAME}")
df = pd.read_csv(CSV_FILE)
df.to_sql("tickets", engine, if_exists="replace", index=False)

print(f"Done! Loaded {len(df)} rows into 'tickets' table.")

