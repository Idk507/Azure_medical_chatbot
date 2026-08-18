from flask import Flask, request, jsonify, send_from_directory, render_template, session, redirect, url_for
from azure.storage.filedatalake import DataLakeServiceClient
import streamlit as st
import json
import uuid
import os
from config import *
import threading
import multiprocessing
import fitz  # PyMuPDF for PDF text extraction
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
import requests
from streamlit_folium import folium_static
from googlesearch import search  # Import the googlesearch library
from geopy.geocoders import Nominatim
from azure.storage.blob import BlobServiceClient  # Import Azure Blob Storage client
from azure.storage.filedatalake import DataLakeServiceClient  # For Azure Data Lake image uploads
from chat import *  # Import the chat function from chat.py
from functools import wraps
import logging
import google.generativeai as genai
from pres import (
    analyze_handwritten_prescription, extract_text_from_image,
    get_current_location, get_nearby_pharmacies, display_azure_map
)

import json
from storage import upload_file_tolake
from config import CONNECTION_STRING, FILE_SYSTEM_NAME

# Initialize Azure Data Lake Storage clients
service_client = DataLakeServiceClient.from_connection_string(CONNECTION_STRING)
file_system_client = service_client.get_file_system_client(FILE_SYSTEM_NAME)


def authenticate_user(username, password):
    try:
        # List all files in users directory to find the correct JSON file
        users_path = "users"
        files = file_system_client.get_paths(users_path)
        user_file = None
        
        # Debug print
        print(f"Attempting login for username: {username}")
        
        # Find the JSON file that starts with the username
        for file in files:
            if file.name.startswith(f"users/{username}_"):
                user_file = file.name
                print(f"Found user file: {user_file}")
                break
                
        if not user_file:
            print("User file not found")
            return False, None
            
        file_client = file_system_client.get_file_client(user_file)
        user_data = json.loads(file_client.download_file().readall())
        
        # Debug print
        print(f"Stored password: {user_data['password']}")
        print(f"Provided password: {password}")
        
        # Compare passwords
        if str(user_data["password"]) == str(password):
            print("Password match successful")
            return True, user_data
            
        print("Password match failed")
        return False, None
    except Exception as e:
        print(f"Error authenticating user: {str(e)}")
        return False, None

def create_user_folder(username, user_id):
    try:
        # Get client for user-data container
        user_container = service_client.get_file_system_client("user-data")
        
        # Create user directory with username_userid format
        folder_name = f"{username}_{user_id}"
        directory_client = user_container.get_directory_client(folder_name)
        directory_client.create_directory()
        
        # Create subdirectories for different types of medical records
        subdirs = ["medical-reports","prescriptions", "lab_reports", "medical_history", "diagnosis"]
        for subdir in subdirs:
            subdir_client = user_container.get_directory_client(f"{folder_name}/{subdir}")
            subdir_client.create_directory()
        return True
    except Exception as e:
        print(f"Error creating user folder: {str(e)}")
        return False

def generate_user_id():
    # Get last ID number from a counter file or generate new
    try:
        counter_file = file_system_client.get_file_client("counter.txt")
        current_num = int(counter_file.download_file().readall())
        new_num = current_num + 1
    except:
        new_num = 1000  # Start from 1000
    
    # Update counter
    counter_file = file_system_client.get_file_client("counter.txt")
    counter_file.upload_data(str(new_num), overwrite=True)
    
    return f"onemeduser{new_num}"

def create_account(username, password, role, email, phone, age):
    try:
        user_id = generate_user_id()
        # Create user identifier combining username and ID
        user_identifier = f"{username}_{user_id}"
        
        # Check if user already exists using new identifier format
        user_directory = f"users/{user_identifier}.json"
        if file_system_client.get_file_client(user_directory).exists():
            return False, "Username already exists"
        
        # Store user data in variables
        user_id_var = user_id
        username_var = username
        password_var = password
        role_var = role
        email_var = email
        phone_var = phone
        age_var = age
        
        # Create user data dictionary for JSON storage
        user_data = {
            "id": user_id_var,
            "username": username_var,
            "password": password_var,
            "role": role_var,
            "email": email_var,
            "phone": phone_var,
            "age": age_var
        }
        
        # Save JSON file with combined identifier
        file_client = file_system_client.get_file_client(user_directory)
        file_client.upload_data(json.dumps(user_data), overwrite=True)

        # Create user folder with same identifier
        if not create_user_folder(username, user_id):
            return False, "Failed to create user storage"
            
        return True, "Account created successfully"
    except Exception as e:
        return False, str(e)