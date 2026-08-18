import streamlit as st
import fitz  # PyMuPDF for PDF text extraction
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
import requests
from dotenv import load_dotenv
import os
load_dotenv()
from streamlit_folium import folium_static
from googlesearch import search  # Import the googlesearch library
from geopy.geocoders import Nominatim
from azure.storage.blob import BlobServiceClient  # Import Azure Blob Storage client
from flask import Flask, render_template, request, jsonify

app = Flask(__name__, template_folder='templates')

# Add Azure configuration at the beginning
AZURE_MAPS_KEY = os.getenv("AZURE_MAPS_KEY")

# Initialize the Blob Service Client
BLOB_SERVICE_CLIENT = BlobServiceClient.from_connection_string(os.getenv("CONNECTION_STRING"))

def upload_to_blob_storage_diag(container_name, directory_name, file_name, file_data):
    """Upload a file to a specific container and directory in Azure Blob Storage."""
    try:
        blob_client = BLOB_SERVICE_CLIENT.get_blob_client(container=container_name, blob=f"{directory_name}/{file_name}")
        blob_client.upload_blob(file_data, overwrite=True)
        st.success(f"File {file_name} uploaded to {container_name}/{directory_name} successfully.")
    except Exception as e:
        st.error(f"Failed to upload file to Azure Blob Storage: {e}")

# ----- Normal Reference Ranges -----
NORMAL_RANGES_diag = {
    "Hemoglobin": (13.0, 16.5),
    "RBC Count": (4.5, 5.5),
    "Hematocrit": (40, 49),
    "MCV": (83, 101),
    "MCH": (27.1, 32.5),
    "MCHC": (32.5, 36.7),
    "RDW CV": (11.6, 14),
    "WBC Count": (4000, 10000),
    "Platelet Count": (150000, 410000),
    "MPV": (7.5, 10.3),
    "ESR": (0, 14),
    "Cholesterol": (0, 200),
    "Triglyceride": (0, 150),
    "HDL Cholesterol": (40, 60),
    "Direct LDL": (0, 100),
    "VLDL": (15, 35),
    "CHOL/HDL Ratio": (0, 5.0),
    "LDL/HDL Ratio": (0, 3.5),
    "Fasting Blood Sugar": (74, 106),
    "HbA1c": (0, 5.7),
    "Mean Blood Glucose": (80, 120),
    "T3 - Triiodothyronine": (0.58, 1.59),
    "T4 - Thyroxine": (4.87, 11.72),
    "TSH - Thyroid Stimulating Hormone": (0.35, 4.94),
    "Microalbumin": (0, 16.7),
    "Total Protein": (6.3, 8.2),
    "Albumin": (3.5, 5.0),
    "Globulin": (2.3, 3.5),
    "A/G Ratio": (1.3, 1.7),
    "Total Bilirubin": (0.2, 1.3),
    "Conjugated Bilirubin": (0.0, 0.3),
    "Unconjugated Bilirubin": (0.0, 1.1),
    "Delta Bilirubin": (0.0, 0.2),
    "Iron": (49, 181),
    "Total Iron Binding Capacity": (261, 462),
    "Transferrin Saturation": (20, 50),
    "Homocysteine, Serum": (6.0, 14.8),
    "Creatinine, Serum": (0.66, 1.25),
    "Urea": (19.3, 43.0),
    "Blood Urea Nitrogen": (9.0, 20.0),
    "Uric Acid": (3.5, 8.5),
    "Calcium": (8.4, 10.2),
    "SGPT": (0, 50),
    "SGOT": (17, 59),
    "Sodium": (136, 145),
    "Potassium": (3.5, 5.1),
    "Chloride": (98, 107),
    "25(OH) Vitamin D": (30, 100),
    "Vitamin B12": (187, 833),
    "IgE": (0, 87),
    "PSA-Prostate Specific Antigen, Total": (0, 40.57)
}

# ----- Logical Groups for Visualizations -----
GROUPS = {
    "Complete Blood Count": ["Hemoglobin", "RBC Count", "Hematocrit", "MCV", "MCH", "MCHC", "RDW CV", "WBC Count", "Platelet Count", "MPV", "ESR"],
    "Lipid Profile": ["Cholesterol", "Triglyceride", "HDL Cholesterol", "Direct LDL", "VLDL", "CHOL/HDL Ratio", "LDL/HDL Ratio"],
    "Blood Sugar": ["Fasting Blood Sugar", "HbA1c", "Mean Blood Glucose"],
    "Thyroid": ["T3 - Triiodothyronine", "T4 - Thyroxine", "TSH - Thyroid Stimulating Hormone"],
    "Urine": ["Microalbumin", "Total Protein", "Albumin", "Globulin", "A/G Ratio", "Total Bilirubin", "Conjugated Bilirubin", "Unconjugated Bilirubin", "Delta Bilirubin"],
    "Iron Studies": ["Iron", "Total Iron Binding Capacity", "Transferrin Saturation"],
    "Renal": ["Creatinine, Serum", "Urea", "Blood Urea Nitrogen", "Uric Acid"],
    "Electrolytes": ["Calcium", "Sodium", "Potassium", "Chloride"],
    "Vitamins": ["25(OH) Vitamin D", "Vitamin B12"],
    "Other": ["IgE", "Homocysteine, Serum", "PSA-Prostate Specific Antigen, Total"]
}

def extract_text_from_pdf_diag(pdf_file):
    """Extract text from a PDF file."""
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = "\n".join([page.get_text("text") for page in doc])
        if not text.strip():
            raise ValueError("No text found in PDF.")
        return text.strip()
    except Exception as e:
        st.error(f"Failed to process PDF: {e}")
        return ""

def extract_medical_values_diag(text):
    """Extract medical parameter values from text."""
    extracted_data = {}
    for parameter, _ in NORMAL_RANGES_diag.items():
        pattern = rf"{re.escape(parameter)}"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            start = match.start()
            window = text[start:start+150]
            numbers = re.findall(r"\d+(?:\.\d+)?", window)
            valid_numbers = [n for n in numbers if re.match(r"^\d+(?:\.\d+)?$", n)]
            if valid_numbers:
                try:
                    if len(valid_numbers) >= 3:
                        measured = float(valid_numbers[-1])
                    elif len(valid_numbers) == 2:
                        measured = float(valid_numbers[1])
                    else:
                        measured = float(valid_numbers[0])
                    extracted_data[parameter] = measured
                except ValueError:
                    st.warning(f"Could not convert value for {parameter} to float.")
    return extracted_data

def plot_medical_report_diag(extracted_data, title="Medical Report Parameters vs Normal Ranges"):
    """Plot medical data against normal ranges."""
    if not extracted_data:
        st.warning("No medical values extracted.")
        return

    labels, values, low_lines, high_lines, colors = [], [], [], [], []
    for parameter, measured in extracted_data.items():
        if parameter in NORMAL_RANGES_diag:
            low, high = NORMAL_RANGES_diag[parameter]
            labels.append(parameter)
            values.append(measured)
            low_lines.append(low)
            high_lines.append(high)
            colors.append("blue" if measured < low else "red" if measured > high else "green")

    y_pos = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, max(6, len(labels)*0.35)))
    ax.barh(y_pos, values, color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Measured Value")
    ax.set_title(title)

    for i in range(len(labels)):
        ax.plot([low_lines[i], low_lines[i]], [i - 0.4, i + 0.4], "gray", linestyle="dashed", linewidth=1)
        ax.plot([high_lines[i], high_lines[i]], [i - 0.4, i + 0.4], "gray", linestyle="dashed", linewidth=1)
        ax.text(values[i], i, f" {values[i]}", va="center", ha="left", fontsize=8)

    st.pyplot(fig)

def get_current_location_diag():
    """Get user's approximate location using an IP-based geolocation service."""
    try:
        response = requests.get("https://ipinfo.io/json").json()
        loc = response["loc"].split(",")  # Extract latitude and longitude
        latitude, longitude = float(loc[0]), float(loc[1])
        return latitude, longitude
    except Exception as e:
        st.error("Error getting location: " + str(e))
        return None

def get_nearby_pharmacies_diag(lat, lon, radius=5000):
    """Find nearby pharmacies using OpenStreetMap (Overpass API)."""
    overpass_url = "http://overpass-api.de/api/interpreter"
    query = f"""
    [out:json];
    node
      ["amenity"="pharmacy"]
      (around:{radius},{lat},{lon});
    out;
    """
    response = requests.get(overpass_url, params={"data": query})
    data = response.json()

    pharmacies = []
    geolocator = Nominatim(user_agent="pharmacy_locator") # Initialize Nominatim
    for element in data.get("elements", []):
        name = element.get("tags", {}).get("name", "Unknown Pharmacy")
        lat, lon = element["lat"], element["lon"]
        try:
            location = geolocator.reverse((lat, lon), exactly_one=True)
            address = location.address if location else "Address not found"
        except Exception as e:
            address = "Address not found"
            st.error(f"Error getting address for {name}: {e}")
        pharmacies.append({"name": name, "address": address, "latitude": lat, "longitude": lon})
    
    return pharmacies

def display_azure_map_diag(lat, lon, pharmacies):
    """Display an Azure Map centered on the given coordinates with pharmacy markers."""
    map_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Azure Maps</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
        <link rel="stylesheet" href="https://atlas.microsoft.com/sdk/javascript/mapcontrol/2/atlas.min.css" type="text/css">
        <script src="https://atlas.microsoft.com/sdk/javascript/mapcontrol/2/atlas.min.js"></script>
        <style>
            #myMap {{
                width: 100%;
                height: 500px;
            }}
            .pinStyle {{
                width: 20px;
                height: 20px;
                border-radius: 50%;
                background-color: red;
                cursor: pointer;
                border: 2px solid white;
            }}
            .userPinStyle {{
                width: 20px;
                height: 20px;
                border-radius: 50%;
                background-color: blue;
                cursor: pointer;
                border: 2px solid white;
            }}
        </style>
        <script>
            function GetMap() {{
                var map = new atlas.Map('myMap', {{
                    center: [{lon}, {lat}],
                    zoom: 14,
                    authOptions: {{
                        authType: 'subscriptionKey',
                        subscriptionKey: '{AZURE_MAPS_KEY}'
                    }}
                }});

                map.events.add('load', function() {{
                    // Add a pin for the user's location
                    var userPin = new atlas.HtmlMarker({{
                        position: [{lon}, {lat}],
                        htmlContent: '<div class="userPinStyle"></div>',
                        popup: new atlas.Popup({{
                            content: 'Your Location',
                            pixelOffset: [0, -20]
                        }})
                    }});
                    map.markers.add(userPin);

                    // Add pins for pharmacies
                    var pharmacyData = {str([{
                            "name": p["name"],
                            "latitude": p["latitude"],
                            "longitude": p["longitude"]
                        } for p in pharmacies]).replace("'", '"')};
                    pharmacyData.forEach(function (pharmacy) {{
                        var googleMapsUrl = 'https://www.google.com/maps/dir/?api=1&destination=' + pharmacy.latitude + ',' + pharmacy.longitude;
                        var pharmacyPin = new atlas.HtmlMarker({{
                            position: [pharmacy.longitude, pharmacy.latitude],
                            htmlContent: '<div class="pinStyle"></div>',
                            popup: new atlas.Popup({{
                                content: pharmacy.name + '<br><a href="' + googleMapsUrl + '" target="_blank">Navigate</a>',
                                pixelOffset: [0, -20]
                            }})
                        }});
                        map.markers.add(pharmacyPin);
                    }});
                }});
            }}
        </script>
    </head>
    <body onload="GetMap()">
        <div id="myMap"></div>
    </body>
    </html>
    """
    st.components.v1.html(map_html, height=500)

def diagnosis_report_analyzer_diag():
    st.write("Upload your lab report (PDF) to visualize parameters against normal ranges.")
    uploaded_file = st.file_uploader("Upload Lab Report (PDF)", type=["pdf"])
    if uploaded_file is not None:
        # Upload the PDF to Azure Blob Storage
        upload_to_blob_storage_diag("your-container-name", "your-directory-name", uploaded_file.name, uploaded_file.getvalue())  # Replace with your actual container and directory names

        text = extract_text_from_pdf_diag(uploaded_file)
        if not text:
            st.error("No text could be extracted from the PDF.")
            return

        with st.expander("Extracted Text Preview"):
            st.text_area("Extracted Text", value=text, height=200)

        medical_values = extract_medical_values_diag(text)
        if medical_values:
            st.subheader("Extracted Medical Values (Cumulative)")
            st.dataframe(pd.DataFrame(list(medical_values.items()), columns=["Parameter", "Measured Value"]))
            st.subheader("Cumulative Visualization")
            plot_medical_report_diag(medical_values)
            st.subheader("Separate Component Visualizations")
            for group_name, parameters in GROUPS.items():
                group_data = {p: medical_values[p] for p in parameters if p in medical_values}
                if group_data:
                    st.markdown(f"{group_name}")
                    st.dataframe(pd.DataFrame(list(group_data.items()), columns=["Parameter", "Measured Value"]))
                    plot_medical_report_diag(group_data, title=f"{group_name}: Parameters vs Normal Ranges")
                else:
                    st.info(f"No data extracted for {group_name}.")
                    
            # --- Nearby Pharmacies Locator ---
            st.markdown("---")
            st.subheader("🏥 Nearby Physical Pharmacies")
            location = get_current_location_diag()
            
            if location:
                lat, lon = location
                st.success(f"Your Current Location: {lat}, {lon}")

                # Get nearby pharmacies
                pharmacies = get_nearby_pharmacies_diag(lat, lon)

                if pharmacies:
                    st.subheader("Nearby Pharmacies")
                    st.dataframe(pharmacies)
                    st.subheader("Pharmacy Locations on Map")
                    display_azure_map_diag(lat, lon, pharmacies) # Use Azure Maps
                else:
                    st.warning("No pharmacies found nearby.")
            else:
                st.error("Could not retrieve your location.")
        else:
            st.warning("No medical parameters could be extracted from the report.")



def main():
    app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == "__main__":
    main()

