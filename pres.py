import streamlit as st
import re
from PIL import Image
import pytesseract
import cv2
import google.generativeai as genai
import io
import uuid
import requests
# import geopandas as gpd # Removed
# import folium # Removed
from streamlit_folium import folium_static
from googlesearch import search  # Import the googlesearch library
import numpy as np # Import numpy
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
import time
from geopy.geocoders import Nominatim
import pandas as pd  # Import pandas
from azure.storage.blob import BlobServiceClient  # Import Azure Blob Storage client
import os
from dotenv import load_dotenv
from config import CONNECTION_STRING, FILE_SYSTEM_NAME
load_dotenv()
from azure.storage.blob import BlobServiceClient  # Import Azure Blob Storage client
from fuzzywuzzy import process  # Add import for fuzzy matching

# Set your Google API key directly
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

# Add Azure configuration after your Google API key setup
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_KEY")

# Update the Azure Translator configuration
TRANSLATOR_ENDPOINT = os.getenv("TRANSLATOR_ENDPOINT")
TRANSLATOR_KEY = os.getenv("TRANSLATOR_KEY")
TRANSLATOR_LOCATION = os.getenv("TRANSLATOR_LOCATION")

# Add this dictionary after your GROUPS dictionary
LANGUAGE_CODES = {
    "English": "en",
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Chinese": "zh",
    "Japanese": "ja",
    "Korean": "ko",
    "Arabic": "ar",
    "Russian": "ru",
    "Bengali": "bn",
    "Tamil": "ta",
    "Telugu": "te",
    "Marathi": "mr",
    "Gujarati": "gu"
}

# Initialize the Computer Vision client
try:
    vision_client = ComputerVisionClient(
        endpoint=AZURE_ENDPOINT,
        credentials=CognitiveServicesCredentials(AZURE_KEY)
    )
except Exception as e:
    vision_client = None
    st.error(f"Failed to initialize Azure Computer Vision: {e}")

# Initialize the Blob Service Client
BLOB_SERVICE_CLIENT = BlobServiceClient.from_connection_string(CONNECTION_STRING)
def upload_to_blob_storage(container_name, directory_name, file_name, file_data):
    """Upload a file to a specific container and directory in Azure Blob Storage."""
    try:
        blob_client = BLOB_SERVICE_CLIENT.get_blob_client(container=container_name, blob=f"{directory_name}/{file_name}")
        blob_client.upload_blob(file_data, overwrite=True)
        st.success(f"File {file_name} uploaded to {container_name}/{directory_name} successfully.")
    except Exception as e:
        st.error(f"Failed to upload file to Azure Blob Storage: {e}")

def correct_medication_names(meds_list, original_text):
    """Auto-correct medicine names using Google Gemini API with context from the original text."""
    if not meds_list:
        return {}
    
    # Configure Google Generative AI with API key
    genai.configure(api_key=GOOGLE_API_KEY)
    
    # Extract just the medication names from the tuples
    med_names = [med[1] for med in meds_list]
    
    prompt = f"""
    I've extracted these potential medication names from a prescription: {', '.join(med_names)}
    
    Original prescription text:
    {original_text}
    
    Please provide the correct standard medication names for each, ensuring that:
    1. Names are properly spelled according to medical standards
    2. Dosage information (if present) is correctly formatted
    3. Each medication is a recognized pharmaceutical product
    
    Return ONLY the corrected names in a comma-separated format, keeping the exact same number of entries as I provided.
    """
    
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        
        # Extract and process the response
        corrected_meds = response.text.strip().split(",")
        corrected_meds = [med.strip() for med in corrected_meds]
        
        # Ensure we have the same number of corrections as original medicines
        if len(corrected_meds) != len(med_names):
            # If counts don't match but we have some corrections, try to map them
            if len(corrected_meds) > 0:
                mapped_corrections = {}
                remaining_originals = med_names.copy()
                remaining_corrections = corrected_meds.copy()
                
                # First pass - exact matches
                for original in med_names.copy():
                    for correction in corrected_meds.copy():
                        # If we find an exact match or close match, map it
                        if original.lower() == correction.lower() or original.lower() in correction.lower() or correction.lower() in original.lower():
                            mapped_corrections[original] = correction
                            if original in remaining_originals:
                                remaining_originals.remove(original)
                            if correction in remaining_corrections:
                                remaining_corrections.remove(correction)
                
                # Second pass - for remaining items, map by similarity
                for original in remaining_originals:
                    if remaining_corrections:  # Only proceed if we have corrections left
                        best_match = min(remaining_corrections, key=lambda x: abs(len(x) - len(original)))
                        mapped_corrections[original] = best_match
                        remaining_corrections.remove(best_match)
                    else:
                        # If we run out of corrections, keep the original
                        mapped_corrections[original] = original
                
                return mapped_corrections
            
            # If we got zero corrections, return originals
            return {med_name: med_name for med_name in med_names}
            
        # If counts match, simple mapping should work
        return dict(zip(med_names, corrected_meds))
    except Exception as e:
        st.error(f"Error correcting medication names: {e}")
        return {med_name: med_name for med_name in med_names}  # Return originals if correction fails

def extract_text_from_image(image_file, target_language="en"):
    """Extract text from an image using Azure Computer Vision or Tesseract as fallback."""
    try:
        # If Azure client is available, try Azure OCR first
        if (vision_client):
            image_bytes = image_file.getvalue()
            read_response = vision_client.read_in_stream(io.BytesIO(image_bytes), raw=True)
            read_operation_location = read_response.headers["Operation-Location"]
            operation_id = read_operation_location.split("/")[-1]

            # Wait for the operation to complete
            while True:
                read_result = vision_client.get_read_result(operation_id)
                if read_result.status not in ['notStarted', 'running']:
                    break
                time.sleep(1)

            # Extract text from Azure results
            azure_text = ""
            if read_result.status == OperationStatusCodes.succeeded:
                for text_result in read_result.analyze_result.read_results:
                    for line in text_result.lines:
                        azure_text += line.text + "\n"
            
            # If Azure OCR fails, fall back to Tesseract
            if not azure_text.strip():
                # Reset file pointer and use PIL/Tesseract
                image_file.seek(0)
                image = Image.open(image_file)
                # Convert to array for OpenCV processing
                image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
                # Apply image preprocessing
                processed_img = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY, 31, 2
                )
                # Extract text using Tesseract
                azure_text = pytesseract.image_to_string(processed_img)

        else:
            # Skip Azure and use Tesseract directly
            image_file.seek(0)
            image = Image.open(image_file)
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
            processed_img = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 31, 2
            )
            azure_text = pytesseract.image_to_string(processed_img)

        # Enhanced pattern to better capture injectable medications and various formats
        med_pattern = r"(?i)(Tab|Tablet|Cap|Capsule|Inj|Injection|Rx|Syp|Syrup|Drp|Drops|Lotion|Cream|Gel|Amp|Ampoule|Vial)\.?\s*([A-Za-z0-9\-\s]+(?:\d+\s*(?:mg|mcg|ml|g|IU|%|units))?)"
        initial_medicines = re.findall(med_pattern, azure_text)

        # Use Gemini to comprehensively identify all medications
        if GOOGLE_API_KEY:
            try:
                genai.configure(api_key=GOOGLE_API_KEY)
                model = genai.GenerativeModel("gemini-1.5-flash")
                
                medicine_prompt = f"""
                You are a medical professional analyzing a prescription.
                
                Original prescription text:
                {azure_text}
                
                Please identify ALL medications mentioned in this prescription with these instructions:
                1. Extract EACH medication name accurately, including brand/generic names
                2. Include dosage information when available (e.g., "500mg")
                3. Identify ALL types of medications including tablets, capsules, injectables, syrups, etc.
                4. Pay special attention to injectable medications that might be listed as Inj, Injection, IV, IM, etc.
                5. Correct any obvious spelling errors in medication names
                6. Ensure you capture combination medications (e.g., "Augmentin Duo")
                
                Format your response as a list with each medication on a new line, following this pattern:
                [Type]: [Medication Name with Dosage]
                
                Examples:
                Tab: Augmentin Duo 625mg
                Cap: Amoxicillin 500mg
                Inj: Asthalin P10
                Injection: Deca-1m
                Medicine: Paracetamol 650mg
                
                If no type is specified, use "Medicine:" as the type.
                """
                
                # Get AI-identified medications
                medicine_response = model.generate_content(medicine_prompt)
                ai_medicines_text = medicine_response.text.strip()
                
                # Process AI response into our format
                ai_medicines = []
                for line in ai_medicines_text.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Extract the medication type and name
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        med_type = parts[0].strip()
                        med_name = parts[1].strip()
                        ai_medicines.append((med_type, med_name))
                
                # Combine initial regex findings with AI findings
                combined_medicines = []
                seen_meds = set()
                
                # Add regex-identified medicines first
                for med_type, med_name in initial_medicines:
                    med_key = f"{med_type.lower()}:{med_name.lower()}"
                    if med_key not in seen_meds:
                        combined_medicines.append((med_type, med_name))
                        seen_meds.add(med_key)
                
                # Add AI-identified medicines that weren't already found
                for med_type, med_name in ai_medicines:
                    med_key = f"{med_type.lower()}:{med_name.lower()}"
                    if med_key not in seen_meds:
                        combined_medicines.append((med_type, med_name))
                        seen_meds.add(med_key)
                
                # If we found medicines, verify and correct them
                if combined_medicines:
                    # Get final verified corrections with full context
                    corrections = correct_medication_names(combined_medicines, azure_text)
                    
                    # Format the output text
                    final_text = (
                        "EXTRACTED TEXT:\n" + 
                        "-" * 50 + "\n" +
                        azure_text + "\n\n" +
                        "RECOGNIZED MEDICINES:\n" +
                        "-" * 50 + "\n"
                    )
                    
                    # Build the medicine lines with corrections
                    med_lines = []
                    corrected_medicines = []
                    
                    for med_type, med_name in combined_medicines:
                        corrected_name = corrections.get(med_name, med_name)
                        correction_note = f" (corrected from '{med_name}')" if corrected_name != med_name else ""
                        med_lines.append(f"🩺 {med_type}: {corrected_name}{correction_note}")
                        corrected_medicines.append((med_type, corrected_name))
                    
                    final_text += "\n".join(med_lines)
                    return final_text.strip(), corrected_medicines
                
            except Exception as e:
                st.warning(f"Advanced medicine detection encountered an issue: {e}")
                # Fall back to basic processing if AI enhancement fails
        
        # Basic processing if AI enhancement failed or is unavailable
        # If no medicines detected by pattern, try one more approach using AI
        if GOOGLE_API_KEY and azure_text.strip():
            try:
                medicine_fallback_prompt = f"""
                This is a medical prescription text. Please extract just the medication names as a comma-separated list.
                If no medications are found, respond with "No medications found".

                Prescription text:
                {azure_text}
                """

                fallback_response = model.generate_content(medicine_fallback_prompt)
                fallback_medicines = fallback_response.text.strip()

                if fallback_medicines and "no medications found" not in fallback_medicines.lower():
                    med_list = [m.strip() for m in fallback_medicines.split(',')]
                    final_text += "\n".join([f"🩺 AI-detected: {med}" for med in med_list])
                    return final_text.strip(), [("AI-detected", med) for med in med_list]
            except Exception:
                pass

        return final_text.strip(), []  # Return empty list if no medicines found

    except Exception as e:
        st.error(f"Error processing image: {e}")
        return "", []  # Return empty list in case of error

def analyze_handwritten_prescription(text, translate_to=None):
    """Analyze and complete handwritten prescription text using AI with optional translation."""
    if not GOOGLE_API_KEY:
        st.error("Google API key is not set. Please set the GOOGLE_API_KEY environment variable.")
        return "AI analysis failed due to missing API key."
    try:
        prompt = f"""
You are a medical assistant. Given the following handwritten prescription text, please:
1. Analyze the text and provide the probable medical condition being treated.
2. Suggest a complete prescription with clear usage instructions if any text seems incomplete or unclear.

Prescription Text:
{text}

Please ensure the text is patient-friendly, correcting any mistakes and completing missing parts where possible.
        """
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        result = response.text if response and response.text else "AI analysis and completion failed. Please try again."
        
        # Translate result if a target language is specified
        if translate_to and result:
            try:
                path = '/translate'
                constructed_url = TRANSLATOR_ENDPOINT + path

                params = {
                    'api-version': '3.0',
                    'from': 'en',
                    'to': [translate_to]  # API expects an array
                }

                headers = {
                    'Ocp-Apim-Subscription-Key': TRANSLATOR_KEY,
                    'Ocp-Apim-Subscription-Region': TRANSLATOR_LOCATION,
                    'Content-type': 'application/json',
                    'X-ClientTraceId': str(uuid.uuid4())
                }

                # Debug logging
                with st.expander("Debug Info"):
                    st.write("Request URL:", constructed_url)
                    st.write("Request Headers:", {k: v for k, v in headers.items() if k != 'Ocp-Apim-Subscription-Key'})
                    st.write("Request Params:", params)

                body = [{
                    'text': result
                }]

                translation_response = requests.post(
                    constructed_url,
                    params=params,
                    headers=headers,
                    json=body,
                    timeout=10
                )

                if translation_response.status_code == 200:
                    translations = translation_response.json()
                    if translations and len(translations) > 0:
                        translated = translations[0]['translations'][0]['text']
                        return translated
                else:
                    st.error(f"Translation failed ({translation_response.status_code}): {translation_response.text}")
                    return result  # Fall back to original text
                    
            except Exception as e:
                st.error(f"Translation error: {str(e)}")
                return result  # Fall back to original text
        
        return result
    except Exception as e:
        st.error(f"Error during AI analysis: {e}")
        return "AI analysis failed due to an error."

def search_online_pharmacies(medicine_name):
    """Search for specific online pharmacy websites for medicines."""
    pharmacy_urls = {
        "MedPlus": f"https://www.medplusmart.com/searchProduct?text={medicine_name}",
        "NetMeds": f"https://www.netmeds.com/catalogsearch/result/{medicine_name}/all",
        "1mg": f"https://www.1mg.com/search/all?name={medicine_name}",
        "PharmEasy": f"https://pharmeasy.in/search/all?name={medicine_name}",
        "Apollo Pharmacy": f"https://www.apollopharmacy.in/search-medicines/{medicine_name}"
    }
    return pharmacy_urls

def get_current_location():
    """Get user's precise location using browser's Geolocation API with fallback to IP-based geolocation."""
    
    # Create an HTML component using the browser's Geolocation API
    geolocation_html = """
    <script>
    // Function to send location data back to Streamlit
    function sendLocationToStreamlit(lat, lon, accuracy) {
        const data = {
            lat: lat,
            lon: lon,
            accuracy: accuracy
        };
        window.parent.postMessage({type: "streamlit:setComponentValue", value: data}, "*");
    }
    
    // Function to check if a point is within a geofence
    function isWithinGeofence(lat, lon, fenceLat, fenceLon, radiusKm) {
        // Calculate distance using the Haversine formula
        const R = 6371; // Earth's radius in km
        const dLat = (fenceLat - lat) * Math.PI / 180;
        const dLon = (fenceLon - lon) * Math.PI / 180;
        const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
                Math.cos(lat * Math.PI / 180) * Math.cos(fenceLat * Math.PI / 180) *
                Math.sin(dLon/2) * Math.sin(dLon/2);
        const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
        const distance = R * c;
        
        return distance <= radiusKm;
    }
    
    // Try to get the user's location
    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(
            // Success callback
            function(position) {
                const lat = position.coords.latitude;
                const lon = position.coords.longitude;
                const accuracy = position.coords.accuracy;
                
                // You can define geofences here (example for India)
                // const indiaGeofence = { lat: 20.5937, lon: 78.9629, radius: 1500 }; // Covers most of India
                
                // Uncomment to apply geofencing logic
                // if (isWithinGeofence(lat, lon, indiaGeofence.lat, indiaGeofence.lon, indiaGeofence.radius)) {
                //     sendLocationToStreamlit(lat, lon, accuracy);
                // } else {
                //     sendLocationToStreamlit(null, null, "Outside permitted area");
                // }
                
                // Just send the location without geofencing
                sendLocationToStreamlit(lat, lon, accuracy);
            },
            // Error callback
            function(error) {
                console.error("Error getting geolocation:", error);
                sendLocationToStreamlit(null, null, error.message);
            },
            // Options
            { 
                enableHighAccuracy: true,
                timeout: 5000,
                maximumAge: 0 
            }
        );
    } else {
        sendLocationToStreamlit(null, null, "Geolocation not supported");
    }
    </script>
    <div>Detecting your location...</div>
    """
    
    # Create a container for the location status
    location_status = st.empty()
    location_status.info("📍 Requesting your precise location...")
    
    # Use Streamlit component to get location
    try:
        # Display the HTML and get the result
        result = st.components.v1.html(geolocation_html, height=0, scrolling=False)
        
        # Check if we got valid location data
        if isinstance(result, dict) and result.get('lat') and result.get('lon'):
            lat, lon = result['lat'], result['lon']
            accuracy = result.get('accuracy', 'unknown')
            location_status.success(f"📍 Location detected! (Accuracy: ±{int(accuracy) if isinstance(accuracy, (int, float)) else '?'} meters)")
            return lat, lon
        elif isinstance(result, dict) and result.get('accuracy'):
            # Show error message
            location_status.error(f"📍 Location error: {result['accuracy']}")
        
        # Wait a bit for the browser to get location
        import time
        time.sleep(2)
        
        # If we're still here, fall back to IP-based geolocation
        location_status.warning("📍 Using approximate location based on your IP address")
        try:
            response = requests.get("https://ipinfo.io/json").json()
            if "loc" in response:
                loc = response["loc"].split(",")
                latitude, longitude = float(loc[0]), float(loc[1])
                return latitude, longitude
            else:
                location_status.error("📍 Could not determine your location")
                return None
        except Exception as e:
            location_status.error(f"📍 Error getting location: {str(e)}")
            return None
            
    except Exception as e:
        st.error(f"Error with geolocation component: {e}")
        
        # Fall back to IP-based geolocation
        try:
            response = requests.get("https://ipinfo.io/json").json()
            loc = response["loc"].split(",")
            latitude, longitude = float(loc[0]), float(loc[1])
            return latitude, longitude
        except Exception as ip_error:
            st.error(f"Error getting location: {ip_error}")
            return None

def get_nearby_pharmacies_google(lat, lon, radius=5000, google_api_key=None):
    """Find nearby pharmacies using Google Places API for more comprehensive results."""
    if not google_api_key:
        google_api_key = GOOGLE_API_KEY
        
    base_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    
    params = {
        "location": f"{lat},{lon}",
        "radius": radius,
        "type": "pharmacy",
        "key": google_api_key,
        "rankby": "distance"  # Sort by distance from user location
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        data = response.json()
        
        pharmacies = []
        if data.get("status") == "OK":
            for place in data.get("results", []):
                location = place.get("geometry", {}).get("location", {})
                pharmacy_lat = location.get("lat")
                pharmacy_lon = location.get("lng")
                
                # Calculate distance using haversine formula
                import math
                R = 6371  # Earth radius in kilometers
                dlat = math.radians(pharmacy_lat - lat)
                dlon = math.radians(pharmacy_lon - lon)
                a = math.sin(dlat/2)**2 + math.cos(math.radians(lat)) * math.cos(math.radians(pharmacy_lat)) * math.sin(dlon/2)**2
                c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
                distance = round(R * c, 2)
                
                # Get additional details for rich information
                pharmacy = {
                    "name": place.get("name", "Unnamed Pharmacy"),
                    "address": place.get("vicinity", "Address unavailable"),
                    "latitude": pharmacy_lat,
                    "longitude": pharmacy_lon,
                    "distance": distance,
                    "walking_time": round(distance * 12),  # ~12 min per km
                    "driving_time": round(distance * 2),   # ~2 min per km
                    "place_id": place.get("place_id"),
                    "rating": place.get("rating"),
                    "source": "Google Places"
                }
                
                # Check if pharmacy is open
                if "opening_hours" in place:
                    pharmacy["open_now"] = place["opening_hours"].get("open_now", False)
                
                pharmacies.append(pharmacy)
                
            # Get more details for the top 5 pharmacies (phone numbers, etc.)
            for i, pharmacy in enumerate(pharmacies[:5]):
                if "place_id" in pharmacy:
                    details = get_place_details(pharmacy["place_id"], google_api_key)
                    if details:
                        pharmacy.update(details)
                        
        return pharmacies
    except Exception as e:
        print(f"Error fetching pharmacies from Google: {e}")
        return []

def get_place_details(place_id, google_api_key=None):
    """Get detailed information about a place from Google Places API."""
    if not google_api_key:
        google_api_key = GOOGLE_API_KEY
        
    base_url = "https://maps.googleapis.com/maps/api/place/details/json"
    
    params = {
        "place_id": place_id,
        "fields": "formatted_phone_number,international_phone_number,website,url",
        "key": google_api_key
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        data = response.json()
        
        if data.get("status") == "OK" and "result" in data:
            result = data["result"]
            return {
                "phone": result.get("formatted_phone_number") or result.get("international_phone_number", ""),
                "website": result.get("website", ""),
                "maps_url": result.get("url", "")
            }
        return {}
    except Exception as e:
        print(f"Error fetching place details: {e}")
        return {}

# Update the main pharmacy search function to use Google first, with OSM as fallback
def get_nearby_pharmacies(lat, lon, radius=5000):
    """Find nearby pharmacies using multiple sources prioritizing Google Places API."""
    # Try Google Places API first (most reliable and comprehensive)
    pharmacies = get_nearby_pharmacies_google(lat, lon, radius)
    
    # If Google API fails or returns no results, fall back to OpenStreetMap
    if not pharmacies:
        # Save the original function's implementation as a fallback
        try:
            # Calculate accurate distance between two points
            def calculate_haversine_distance(lat1, lon1, lat2, lon2):
                """Calculate the great circle distance between two points in kilometers."""
                # Convert coordinates from degrees to radians
                import math
                lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
                
                # Haversine formula
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
                c = 2 * math.asin(math.sqrt(a))
                r = 6371  # Radius of Earth in kilometers
                distance = round(r * c, 2)  # Round to 2 decimal places
                return distance
            
            # Original OSM implementation
            overpass_url = "http://overpass-api.de/api/interpreter"
            query = f"""
            [out:json];
            (
              node["amenity"="pharmacy"](around:{radius},{lat},{lon});
              way["amenity"="pharmacy"](around:{radius},{lat},{lon});
              relation["amenity"="pharmacy"](around:{radius},{lat},{lon});
            );
            out center;
            """
            response = requests.get(overpass_url, params={"data": query}, timeout=10)
            data = response.json()
            
            geolocator = Nominatim(user_agent="pharmacy_locator")
            
            osm_pharmacies = []
            for element in data.get("elements", []):
                # Extract pharmacy data from OSM
                # Handling different node types from OSM
                if element["type"] == "node":
                    pharmacy_lat, pharmacy_lon = element["lat"], element["lon"]
                elif "center" in element:
                    pharmacy_lat, pharmacy_lon = element["center"]["lat"], element["center"]["lon"]
                else:
                    continue
                    
                # Get basic info
                name = element.get("tags", {}).get("name", "Unnamed Pharmacy")
                if not name or name == "Unnamed Pharmacy":
                    name = element.get("tags", {}).get("brand", "Pharmacy")
                
                # Calculate distance
                distance = calculate_haversine_distance(lat, lon, pharmacy_lat, pharmacy_lon)
                
                # Get address
                try:
                    location = geolocator.reverse((pharmacy_lat, pharmacy_lon), exactly_one=True, timeout=5)
                    address = location.address if location else "Address unavailable"
                except Exception as e:
                    address = "Address unavailable"
                
                # Get phone number if available
                phone = element.get("tags", {}).get("phone", "") or element.get("tags", {}).get("contact:phone", "")
                
                osm_pharmacies.append({
                    "name": name,
                    "address": address,
                    "latitude": pharmacy_lat,
                    "longitude": pharmacy_lon,
                    "distance": distance,
                    "phone": phone,
                    "source": "OpenStreetMap",
                    "walking_time": round(distance * 12),  # ~12 min per km
                    "driving_time": round(distance * 2)   # ~2 min per km
                })
            
            # Sort by distance
            osm_pharmacies.sort(key=lambda x: x.get("distance", float("inf")))
            pharmacies = osm_pharmacies
        except Exception as e:
            print(f"Error with OSM fallback: {e}")
            # Leave pharmacies as empty list
    
    return pharmacies

def create_map(user_lat, user_lon, pharmacies):
    """Generate an interactive map with Folium."""
    map_ = folium.Map(location=[user_lat, user_lon], zoom_start=14)

    # Add user location marker
    folium.Marker([user_lat, user_lon], 
                popup="Your Location", 
                icon=folium.Icon(color="blue", icon="user")).add_to(map_)

    # Add pharmacy markers
    for pharmacy in pharmacies:
        folium.Marker(
            [pharmacy["latitude"], pharmacy["longitude"]],
            popup=pharmacy["name"],
            icon=folium.Icon(color="red", icon="plus-sign"),
        ).add_to(map_)

    return map_

def display_azure_map(lat, lon, pharmacies, user_accuracy=None):
    """
    Returns structured data for Azure Maps display.
    For Flask integration, also accepts a return_data=True parameter
    to return the structured data instead of HTML.
    """
    azure_maps_key = os.getenv("AZURE_MAPS_KEY")
    
    # Create structured map data for frontend use
    map_data = {
        'apiKey': azure_maps_key,
        'center': {'latitude': lat, 'longitude': lon},
        'pharmacies': pharmacies,
        'accuracy': user_accuracy
    }
    
    # For Flask integration, we can just return this data
    if 'return_data' in locals() and return_data:
        return map_data
        
    # For Streamlit, continue with the existing HTML generation
    map_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Pharmacy Map</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
        <!-- Azure Maps CSS and JS -->
        <link rel="stylesheet" href="https://atlas.microsoft.com/sdk/javascript/mapcontrol/2/atlas.min.css" type="text/css">
        <script src="https://atlas.microsoft.com/sdk/javascript/mapcontrol/2/atlas.min.js"></script>
        <style>
            #mapContainer {{
                width: 100%;
                height: 500px;
                position: relative;
            }}
            .pin {{
                width: 24px;
                height: 24px;
                border-radius: 50%;
                cursor: pointer;
                box-shadow: 0 0 5px rgba(0,0,0,0.5);
            }}
            .pin.user {{
                background-color: blue;
                border: 2px solid white;
            }}
            .pin.pharmacy {{
                background-color: red;
                border: 2px solid white;
            }}
            .map-legend {{
                position: absolute;
                bottom: 20px;
                left: 20px;
                background-color: white;
                padding: 10px;
                border-radius: 5px;
                box-shadow: 0 0 10px rgba(0,0,0,0.2);
                z-index: 1000;
                font-family: Arial, sans-serif;
                font-size: 14px;
            }}
            .legend-item {{
                display: flex;
                align-items: center;
                margin: 5px 0;
            }}
            .legend-color {{
                width: 16px;
                height: 16px;
                border-radius: 50%;
                margin-right: 8px;
            }}
            .legend-label {{
                font-weight: 500;
            }}
        </style>
    </head>
    <body>
        <div id="mapContainer">
            <div id="map" style="width:100%; height:100%;"></div>
            <!-- Map legend -->
            <div class="map-legend">
                <div class="legend-title" style="font-weight:bold; margin-bottom:8px;">Map Legend</div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color:blue; border:1px solid white;"></div>
                    <div class="legend-label">Your Location</div>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color:red; border:1px solid white;"></div>
                    <div class="legend-label">Pharmacy</div>
                </div>
            </div>
        </div>
        
        <script>
            // Wait for the page to load before creating the map
            window.onload = function() {{
                try {{
                    // Create the map instance
                    var map = new atlas.Map('map', {{
                        center: [{lon}, {lat}],
                        zoom: 14,
                        authOptions: {{
                            authType: 'subscriptionKey',
                            subscriptionKey: '{azure_maps_key}'
                        }},
                        style: 'road'
                    }});
                    
                    map.events.add('ready', function() {{
                        // Create data source for map objects
                        var dataSource = new atlas.source.DataSource();
                        map.sources.add(dataSource);
                        
                        // Add user location marker
                        var userMarker = new atlas.HtmlMarker({{
                            htmlContent: '<div class="pin user"></div>',
                            position: [{lon}, {lat}],
                            pixelOffset: [0, 0]
                        }});
                        
                        // Create a popup for the user marker
                        var userPopup = new atlas.Popup({{
                            content: '<div style="padding:10px;"><strong>Your Location</strong></div>',
                            position: [{lon}, {lat}],
                            pixelOffset: [0, -15]
                        }});
                        
                        // Add the user marker to the map
                        map.markers.add(userMarker);
                        
                        // Add event listener to show popup when marker is clicked
                        map.events.add('click', userMarker, function() {{
                            userPopup.open(map);
                        }});
                        
                        // Add accuracy circle if available
                        if ({user_accuracy if user_accuracy and isinstance(user_accuracy, (int, float)) else 'null'}) {{
                            // Add accuracy circle as a polygon
                            var accuracyRadius = {user_accuracy if user_accuracy and isinstance(user_accuracy, (int, float)) else 0};
                            
                            // Create circle based on radius in meters
                            var points = [];
                            var numPoints = 36; // Number of points to create circle
                            
                            for (var i = 0; i < numPoints; i++) {{
                                var angle = (i / numPoints) * 2 * Math.PI;
                                var dx = accuracyRadius * Math.cos(angle);
                                var dy = accuracyRadius * Math.sin(angle);
                                
                                // Convert dx/dy in meters to lat/lon degrees
                                // 111,320 meters per degree of latitude
                                // 111,320 * cos(latitude) meters per degree of longitude
                                var lat_m = {lat} + (dy / 111320);
                                var lon_m = {lon} + (dx / (111320 * Math.cos({lat} * Math.PI / 180)));
                                
                                points.push([lon_m, lat_m]);
                            }}
                            
                            // Create polygon from points
                            dataSource.add(new atlas.data.Polygon([points]));
                            
                            // Add layer for the polygon
                            map.layers.add(new atlas.layer.PolygonLayer(dataSource, null, {{
                                fillColor: 'rgba(0, 0, 255, 0.2)',
                                fillOpacity: 0.5
                            }}));
                        }}
                        
                        // Add pharmacy markers
                        var pharmacies = {pharmacies_json};
                        
                        pharmacies.forEach(function(pharmacy) {{
                            // Create marker for pharmacy
                            var pharmacyMarker = new atlas.HtmlMarker({{
                                htmlContent: '<div class="pin pharmacy"></div>',
                                position: [pharmacy.longitude, pharmacy.latitude],
                                pixelOffset: [0, 0]
                            }});
                            
                            // Create popup for pharmacy
                            var pharmacyPopup = new atlas.Popup({{
                                content: '<div style="padding:10px;"><strong>' + pharmacy.name + '</strong><br/>' + 
                                         pharmacy.address + '<br/><br/>' +
                                         '<a href="https://www.google.com/maps/dir/?api=1&destination=' + 
                                         pharmacy.latitude + ',' + pharmacy.longitude + 
                                         '" target="_blank">Get Directions</a></div>',
                                position: [pharmacy.longitude, pharmacy.latitude],
                                pixelOffset: [0, -15]
                            }});
                            
                            // Add marker to map
                            map.markers.add(pharmacyMarker);
                            
                            // Add event listener for marker click
                            map.events.add('click', pharmacyMarker, function() {{
                                pharmacyPopup.open(map);
                            }});
                        }});
                    }});
                }} catch (error) {{
                    console.error("Error creating map:", error);
                    document.getElementById('mapContainer').innerHTML = '<div style="color:red; padding:20px;">Error loading map: ' + error.message + '</div>';
                }}
            }};
        </script>
    </body>
    </html>
    """
    
    # Display the map
    try:
        st.components.v1.html(map_html, height=520, scrolling=False)
    except Exception as e:
        st.error(f"Error displaying map: {str(e)}")
        st.code(map_html, language="html")  # Show the HTML for debugging

def generate_medicines_database():
    """Generate a CSV database of common medications using AI or a fallback list."""
    csv_path = os.path.join(os.path.dirname(__file__), "medication.csv")
    
    # If the file already exists, just return the path
    if (os.path.exists(csv_path)):
        return csv_path
        
    try:
        # Configure Google Generative AI with API key
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        prompt = """
        Generate a comprehensive list of 200 common prescription medications with their correct spelling.
        Include generic and brand names across different therapeutic categories.
        Format the output as CSV with columns: id,name,category,type
        
        Example:
        1,Metformin,Antidiabetic,Oral
        2,Lisinopril,Antihypertensive,Oral
        3,Augmentin,Antibiotic,Oral
        """
        
        response = model.generate_content(prompt)
        csv_content = response.text.strip()
        
        # Save the CSV content
        with open(csv_path, 'w') as f:
            f.write("id,name,category,type\n" + csv_content)
            
        return csv_path
        
    except Exception as e:
        st.warning(f"Could not generate medicines database: {e}")
        
        # Fallback: Create a minimal CSV with common medications
        fallback_meds = [
            "1,Augmentin,Antibiotic,Oral",
            "2,Amoxicillin,Antibiotic,Oral",
            "3,Azithromycin,Antibiotic,Oral",
            "4,Paracetamol,Analgesic,Oral",
            "5,Ibuprofen,NSAID,Oral",
            "6,Metformin,Antidiabetic,Oral",
            "7,Lisinopril,Antihypertensive,Oral",
            "8,Atorvastatin,Cholesterol Lowering,Oral",
            "9,Omeprazole,Proton Pump Inhibitor,Oral",
            "10,Loratadine,Antihistamine,Oral"
        ]
        
        with open(csv_path, 'w') as f:
            f.write("id,name,category,type\n")
            for med in fallback_meds:
                f.write(med + "\n")
                
        return csv_path

def correct_medication_name_fuzzy(misspelled_name):
    """Correct medication names using fuzzy matching against a medicines database."""
    try:
        # Get medicines database path
        csv_path = generate_medicines_database()
        
        # Load the structured dataset (CSV)
        df = pd.read_csv(csv_path)
        
        # Extract list of medicine names
        choices = df["name"].tolist()
        
        # Find best match using fuzzy matching
        best_match, score = process.extractOne(misspelled_name, choices)
        
        # Return best match if score is above threshold
        return best_match if score > 80 else misspelled_name
    
    except Exception as e:
        st.warning(f"Fuzzy matching failed: {e}")
        return misspelled_name  # Return original if error occurs

def final_medication_verification(medications):
    """Performs a final verification of medication names using AI and fuzzy matching as fallback."""
    if not medications:
        return []
    
    # Extract just the medication names
    med_names = [med[1] for med in medications]
    
    # Try online AI verification first
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            "Here is a list of potentially misspelled medicine names. "
            "Please correct them to their proper medical names:\n"
            + ", ".join(med_names)
            + "\nProvide ONLY the corrected names in a comma-separated format with exactly the same number of entries. "
            "If a name is already correct, repeat it in your response."
        )
        
        response = model.generate_content(prompt)
        
        # Extract and process the response - handle various comma formats
        corrected_text = response.text.strip()
        # Remove any markdown code block formatting if present
        if "```" in corrected_text:
            corrected_text = corrected_text.split("```")[1] if len(corrected_text.split("```")) > 1 else corrected_text
            corrected_text = corrected_text.strip()
        
        # Split by comma, handling both comma+space and just comma formats
        corrected_meds = [med.strip() for med in re.split(r',\s*', corrected_text)]
        
        # Create a verified medications list
        
        verified_medications = []
        
        # Ensure we have the same number of corrections as original medicines
        if len(corrected_meds) == len(med_names):
            # Simple 1:1 mapping
            for i, (med_type, med_name) in enumerate(medications):
                verified_medications.append((med_type, corrected_meds[i]))
            return verified_medications
        else:
            # Try to map corrections to original names based on similarity
            for med_type, med_name in medications:
                # Find the closest match in the corrected medicines
                best_match = ""
                highest_similarity = 0
                
                for corrected in corrected_meds:
                    # Calculate similarity (simple case insensitive substring check)
                    similarity = 0
                    if corrected.lower() in med_name.lower() or med_name.lower() in corrected.lower():
                        similarity = len(corrected) / max(len(med_name), len(corrected))
                    
                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        best_match = corrected
                
                # If we found a good match, use it; otherwise keep original
                if highest_similarity > 0.5 and best_match:
                    verified_medications.append((med_type, best_match))
                    # Remove the matched correction to avoid duplicates
                    if best_match in corrected_meds:
                        corrected_meds.remove(best_match)
                else:
                    # Fall back to fuzzy matching for this specific medicine
                    corrected_name = correct_medication_name_fuzzy(med_name)
                    verified_medications.append((med_type, corrected_name))
            
            return verified_medications
            
    except Exception as e:
        st.warning(f"AI verification failed: {e}. Trying fuzzy matching...")
    
    # Fallback to fuzzy matching if AI correction failed
    verified_medications = []
    for med_type, med_name in medications:
        # Use fuzzy matching to correct the name
        corrected_name = correct_medication_name_fuzzy(med_name)
        verified_medications.append((med_type, corrected_name))
    
    return verified_medications

def predict_disease_from_medications(medications):
    """Predict possible diseases or conditions based on the identified medications."""
    if not medications:
        return None
    
    # Extract medication names (without the type)
    med_names = [med_name for _, med_name in medications]
    
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        prompt = f"""
        As a clinical pharmacologist, analyze these medications and predict the most likely medical conditions being treated:
        {', '.join(med_names)}
        
        Provide your response in this JSON format:
        {{
            "primary_condition": "Most likely condition",
            "confidence": "High/Medium/Low",
            "alternative_conditions": ["Possible condition 1", "Possible condition 2"],
            "reasoning": "Brief explanation of your analysis"
        }}
        
        Only respond with the JSON, no additional text.
        """
        
        response = model.generate_content(prompt)
        result_text = response.text.strip()
        
        # Check if the response starts/ends with ```json and ``` and remove them
        if result_text.startswith("```json"):
            result_text = result_text.split("```json")[1]
        if result_text.endswith("```"):
            result_text = result_text.split("```")[0]
        
        # Try to parse as JSON
        import json
        try:
            result = json.loads(result_text.strip())
            return result
        except json.JSONDecodeError:
            # If not valid JSON, return a formatted dictionary
            return {
                "primary_condition": "Condition analysis unavailable",
                "confidence": "Low",
                "alternative_conditions": [],
                "reasoning": "Unable to analyze the medication combination accurately."
            }
    
    except Exception as e:
        st.warning(f"Disease prediction failed: {e}")
        return None

def handwritten_prescription_analyzer():
    st.title("Prescription Reader")
    st.write("Upload an image of a handwritten prescription.")
    
    # Add prescription disclaimer
    st.info("""
    **IMPORTANT DISCLAIMER**: This tool is for educational purposes only. 
    Actual prescriptions must be written by licensed physicians after proper examination and diagnosis. 
    The medications shown are examples and dosages may require clarification from a healthcare provider.
    """)
    
    # Configure Google Generative AI API key
    genai.configure(api_key=GOOGLE_API_KEY)
    
    # Add language selection
    target_language = st.selectbox(
        "Select Translation Language",
        options=list(LANGUAGE_CODES.keys()),
        index=0
    )
    
    image_file = st.file_uploader("Upload Prescription (Image)", type=["jpg", "jpeg", "png"])
    if image_file is not None:
        # Upload the image to Azure Blob Storage
        upload_to_blob_storage("your-container-name", "your-directory-name", image_file.name, image_file.getvalue())  # Replace with your actual container and directory names

        ocr_text, medicines = extract_text_from_image(  # Capture medicines list
            image_file, 
            target_language="en" # Set default language
        )
        if not ocr_text:
            st.error("No text could be extracted from the image.")
        elif len(ocr_text) < 20:
            st.warning("Extracted text is very short. The image might not be clear.")
        else:
            with st.expander("Extracted Prescription Text"):
                st.text_area("OCR Extracted Text", value=ocr_text, height=200)
            st.info("Analyzing handwritten prescription...")
            analysis_result = analyze_handwritten_prescription(ocr_text, translate_to=LANGUAGE_CODES[target_language])
            st.subheader("Handwritten Prescription Analysis")
            st.write(analysis_result)

            # --- Pharmacy Information ---
            if medicines:
                # Apply the final verification to medications using combined approach
                verified_medicines = final_medication_verification(medicines)
                
                # Display a comparison of original vs. verified medicines
                with st.expander("Medication Verification Details"):
                    st.write("### Original vs. Verified Medications")
                    st.write("Corrections are made using AI APIs and fuzzy matching against a medication database.")
                    for (orig_type, orig_name), (ver_type, ver_name) in zip(medicines, verified_medicines):
                        if orig_name != ver_name:
                            st.write(f"✓ {orig_type}: {orig_name} → {ver_name}")
                        else:
                            st.write(f"✓ {orig_type}: {orig_name} (verified)")
                
                # Predict potential disease based on medications
                disease_prediction = predict_disease_from_medications(verified_medicines)
                if disease_prediction:
                    st.subheader("🔍 Potential Condition Analysis")
                    
                    # Display prediction with confidence indicator
                    confidence = disease_prediction.get("confidence", "Low")
                    confidence_color = {
                        "High": "green",
                        "Medium": "orange",
                        "Low": "red"
                    }.get(confidence, "gray")
                    
                    st.markdown(f"""
                    <div style="padding:10px; border-radius:5px; border:1px solid {confidence_color};">
                        <h4>Primary condition: {disease_prediction.get('primary_condition', 'Unknown')}</h4>
                        <p><b>Confidence:</b> <span style="color:{confidence_color};">{confidence}</span></p>
                        <p><b>Reasoning:</b> {disease_prediction.get('reasoning', 'No explanation available')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show alternative conditions if available
                    alternatives = disease_prediction.get('alternative_conditions', [])
                    if alternatives:
                        with st.expander("Alternative possible conditions"):
                            for condition in alternatives:
                                st.write(f"• {condition}")
                    
                    st.info("Note: This analysis is based solely on the medications and should not be considered a diagnosis. Always consult a qualified healthcare professional.")
                
                st.subheader("Recognized Medications")
                
                # Remove duplicate medications before creating the dataframe
                unique_medicines = []
                seen_med_names = set()
                
                for med_type, med_name in verified_medicines:
                    # Check if this medicine name has been seen before
                    if med_name.lower() not in seen_med_names:
                        unique_medicines.append((med_type, med_name))
                        seen_med_names.add(med_name.lower())
                
                # Create a more robust table showing all medications including injectables (using deduplicated list)
                med_data = {
                    "Type": [med_type for med_type, _ in unique_medicines],
                    "Medication Name": [med_name for _, med_name in unique_medicines],
                    "Form": [
                        "Injectable" if med_type.lower() in ["inj", "injection", "amp", "ampoule", "vial"] 
                        else ("Oral" if med_type.lower() in ["tab", "tablet", "cap", "capsule", "syp", "syrup"] 
                        else "Topical" if med_type.lower() in ["cream", "gel", "lotion", "ointment"]
                        else "Other") 
                        for med_type, _ in unique_medicines
                    ]
                }
                
                # Display the medicines in a styled dataframe
                med_df = pd.DataFrame(med_data)
                st.dataframe(
                    med_df,
                    use_container_width=True,
                    column_config={
                        "Medication Name": st.column_config.TextColumn(
                            "Medication Name",
                            help="Verified medication name",
                            width="large"
                        ),
                        "Type": st.column_config.TextColumn("Type", width="small"),
                        "Form": st.column_config.TextColumn("Form", width="medium")
                    }
                )
                
                # If no medicines were actually displayed, show an error
                if len(med_df) == 0:
                    st.error("No medications could be displayed. Please check the image quality.")
                elif len(med_df) != len(verified_medicines):
                    st.warning(f"Only {len(med_df)} out of {len(verified_medicines)} medications could be displayed.")

                st.subheader("🏪 Buy Medicines Online")
                
                # Add an "All Medications" option at the beginning
                tab_options = ["All Medications"] + [f"{med_type}: {med_name}" for med_type, med_name in unique_medicines]
                med_tabs = st.tabs(tab_options)
                
                # Add "All Medications" tab content
                with med_tabs[0]:
                    st.write("### Quick Purchase Options for All Medications")
                    
                    # Create a container for each medicine with all pharmacy options
                    for med_type, med_name in unique_medicines:
                        st.write(f"#### {med_name} ({med_type})")
                        online_pharmacies = search_online_pharmacies(med_name)
                        
                        # Display pharmacy options in a grid for each medicine
                        cols = st.columns(len(online_pharmacies))
                        for idx, (pharmacy_name, url) in enumerate(online_pharmacies.items()):
                            with cols[idx]:
                                st.link_button(f"{pharmacy_name}", url)
                        
                        st.markdown("---")
                
                # Individual medicine tabs (offset by 1 because of "All Medications" tab)
                for i, (med_type, med_name) in enumerate(unique_medicines):
                    with med_tabs[i+1]:
                        st.write(f"### {med_name}")
                        online_pharmacies = search_online_pharmacies(med_name)
                        
                        # Display pharmacy options in a grid
                        cols = st.columns(len(online_pharmacies))
                        for idx, (pharmacy_name, url) in enumerate(online_pharmacies.items()):
                            with cols[idx]:
                                st.link_button(f"Buy on {pharmacy_name}", url)
                                st.caption(f"Search {med_name} on {pharmacy_name}")

                # --- Nearby Pharmacies Locator ---
                st.markdown("---")
                st.subheader("🏥 Nearby Physical Pharmacies")
                
                location_result = get_current_location()
                
                if location_result:
                    if isinstance(location_result, tuple) and len(location_result) == 2:
                        lat, lon = location_result
                        accuracy = None  # Default if not provided
                    elif isinstance(location_result, tuple) and len(location_result) == 3:
                        lat, lon, accuracy = location_result
                    else:
                        lat, lon = location_result
                        accuracy = None
                        
                    st.success(f"Your Current Location: {lat:.6f}, {lon:.6f}")
                    if accuracy and isinstance(accuracy, (int, float)):
                        st.info(f"Location Accuracy: ±{accuracy} meters")

                    # Get nearby pharmacies with a smaller radius for more accurate results
                    search_radius = 3000  # 3 km
                    if accuracy and accuracy < 100:
                        # If we have high accuracy, we can use a smaller radius
                        search_radius = 2000  # 2 km
                        
                    pharmacies = get_nearby_pharmacies(lat, lon, radius=search_radius)

                    if pharmacies:
                        st.subheader(f"Nearby Pharmacies (within {search_radius/1000:.1f} km)")
                        pharmacies_df = pd.DataFrame(pharmacies)  # Convert list of dictionaries to DataFrame
                        st.dataframe(pharmacies_df[["name", "address"]])

                        st.subheader("Pharmacy Locations on Map")
                        
                        # Add a fallback option in case the map doesn't work
                        map_tabs = st.tabs(["Azure Map", "Pharmacy Links"])
                        
                        with map_tabs[0]:
                            display_azure_map(lat, lon, pharmacies, user_accuracy=accuracy)
                        
                        with map_tabs[1]:
                            st.write("### Direct Links to Pharmacies")
                            for i, pharmacy in enumerate(pharmacies):
                                google_maps_url = f"https://www.google.com/maps/dir/?api=1&destination={pharmacy['latitude']},{pharmacy['longitude']}"
                                st.markdown(f"**{i+1}. {pharmacy['name']}**")
                                st.write(f"Address: {pharmacy['address']}")
                                st.link_button(f"Navigate to {pharmacy['name']}", google_maps_url)
                                st.markdown("---")
                    else:
                        st.warning(f"No pharmacies found within {search_radius/1000:.1f} km of your location.")
                else:
                    st.error("Could not retrieve your location. Please allow location access in your browser.")
            else:
                st.info("No medicines recognized in the prescription.")

# ----- Main App -----
def main():
    handwritten_prescription_analyzer() # Call analyzer function

if __name__ == "__main__":
    main()
