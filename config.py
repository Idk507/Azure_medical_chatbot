import os
from dotenv import load_dotenv
from azure.storage.filedatalake import DataLakeServiceClient
from azure.storage.blob import BlobServiceClient

load_dotenv()

STORAGE_ACCOUNT_NAME = os.getenv("STORAGE_ACCOUNT_NAME")
FILE_SYSTEM_NAME = os.getenv("FILE_SYSTEM_NAME")
CONNECTION_STRING = os.getenv("CONNECTION_STRING")
AZURE_MAPS_KEY = os.getenv("AZURE_MAPS_KEY")
azure_maps_key = AZURE_MAPS_KEY
BLOB_SERVICE_CLIENT = BlobServiceClient.from_connection_string(CONNECTION_STRING)
Gemini_API_KEY = os.getenv("GEMINI_API_KEY")

AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_KEY")

TRANSLATOR_ENDPOINT = os.getenv("TRANSLATOR_ENDPOINT")
TRANSLATOR_KEY = os.getenv("TRANSLATOR_KEY")
TRANSLATOR_LOCATION = os.getenv("TRANSLATOR_LOCATION")
