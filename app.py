from flask import Flask, request, jsonify, send_from_directory, render_template, session, redirect, url_for, abort
import os
from pres import (
    analyze_handwritten_prescription, extract_text_from_image,
    get_current_location, get_nearby_pharmacies, display_azure_map,
    predict_disease_from_medications, final_medication_verification,
    search_online_pharmacies, LANGUAGE_CODES
)
import io
import logging
import google.generativeai as genai
import os
import json
import requests
from diagnosis import (NORMAL_RANGES_diag,upload_to_blob_storage_diag,extract_text_from_pdf_diag,extract_medical_values_diag,plot_medical_report_diag,get_current_location_diag,get_nearby_pharmacies_diag,display_azure_map_diag)
try:
    from werkzeug.wrappers import Response as BaseResponse
except ImportError:
    from werkzeug.wrappers.response import Response as BaseResponse
from PIL import Image
from auth import (create_account,generate_user_id,authenticate_user,create_user_folder)
import tempfile
from werkzeug.utils import secure_filename
from tools import *
from config import *
from storage import upload_file_tolake
from functools import wraps  # Add this import for the wraps decorator
from chat import ( 
    load_medical_datasets,
    get_vectorizer,
    search_all_datasets,
    analyze_document,
    load_unstructured_data,
    search_unstructured_data,
    search_medical_websites,
    get_enhanced_response,
    is_medical_query,
    is_disease_query,
    is_non_medical_query,
    extract_medical_content,
    get_non_medical_response,
    get_symptom_questions,
    get_disease_specific_options,
    format_medical_response,
    get_symptom_options,
    get_common_symptoms,
    search_clinical_conditions,
    get_default_symptom_questions
)

from x_ray import MedicalImageAnalyzer_x_ray

GOOGLE_API_KEY = Gemini_API_KEY

service_client = DataLakeServiceClient.from_connection_string(CONNECTION_STRING)
file_system_client = service_client.get_file_system_client(FILE_SYSTEM_NAME)

# Initialize Flask with correct folder configuration
app = Flask(__name__, 
            static_folder='static', 
            template_folder='templates')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
genai.configure(api_key=GOOGLE_API_KEY)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16 MB max file size
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), 'templates'), exist_ok=True)

app.secret_key = os.getenv('FLASK_SECRET_KEY', 'onemed_secret_key')

# Initialize medical datasets and unstructured documents
medical_datasets = load_medical_datasets()
unstructured_documents = load_unstructured_data()

# Debug info route
@app.route('/debug')
def debug_info():
    template_folder = app.template_folder
    static_folder = app.static_folder
    templates_exists = os.path.exists(template_folder)
    index_exists = os.path.exists(os.path.join(template_folder, 'index.html'))
    return jsonify({
        'template_folder': template_folder,
        'static_folder': static_folder,
        'templates_exists': templates_exists,
        'index_html_exists': index_exists,
        'working_directory': os.getcwd()
    })

@app.errorhandler(404)
def not_found_error(error):
    return "The requested URL was not found on the server. If you entered the URL manually please check your spelling and try again.", 404

@app.route('/index', methods=['GET', 'POST'])
def index():
    # Get username from session if available
    username = session.get('username', 'Guest')
    return render_template('index.html', username=username)

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    app.logger.info("Contact route accessed")
    try:
        if request.method == 'POST':
            # Handle form submission
            name = request.form.get('name')
            email = request.form.get('email')
            subject = request.form.get('subject')
            message = request.form.get('message')
            
            app.logger.info(f"Contact form submission: {name}, {email}, {subject}")
            
            # Return with success flag
            return render_template('contact.html', success=True)
        
        return render_template('contact.html')
    except Exception as e:
        app.logger.error(f"Error in contact route: {str(e)}")
        return f"Error: {str(e)}", 500

# Create simple templates for routes to prevent 500 errors

@app.route('/medilocker')
def medilocker():
    # Use this function to run the Streamlit UI in an iframe or redirect
    try:
        # Option 1: Run Streamlit as a subprocess
        import subprocess
        streamlit_port = 8501  # Default Streamlit port
        subprocess.Popen(["streamlit", "run", "medilocker_ui.py", "--server.port", str(streamlit_port)])
        
        # Render a page that embeds the Streamlit UI in an iframe
        return render_template('medilocker.html', streamlit_url=f"http://localhost:{streamlit_port}")
    except Exception as e:
        app.logger.error(f"Error launching Streamlit: {str(e)}")
        return render_template('construction.html', page_title="MediLocker", 
                             error_message="Error launching MediLocker interface.")

@app.route('/about')
def about():
    return render_template('about.html')

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('serve_login'))
        return f(*args, **kwargs)
    return decorated_function


@app.route('/')
def serve_login():
    return render_template('login.html')

# Update the route for login page to handle both GET and POST methods
@app.route('/login', methods=['GET', 'POST'])
def login():  # Changed from login_api to login
    if request.method == 'POST':
        try:
            data = request.form
            username = data.get('username')
            password = data.get('password')
            
            print(f"Login attempt - Username: {username}")
            
            success, user_data = authenticate_user(username, password)
            
            if success and user_data:
                session['username'] = user_data['username']
                session['role'] = user_data['role']
                return jsonify({
                    'success': True,
                    'role': user_data['role'],
                    'redirect': url_for('index')  # Use Flask's url_for to generate correct URL
                })
            
            print("Authentication failed")
            return jsonify({
                'success': False,
                'message': 'Invalid username or password'
            }), 401
            
        except Exception as e:
            print(f"Login error: {str(e)}")
            return jsonify({
                'success': False,
                'message': 'An error occurred during login'
            }), 500
    else:
        # GET request - serve the login page
        return render_template('login.html')

@app.route('/register', methods=['POST'])
def register_api():
    data = request.form
    success, message = create_account(
        data.get('username'),
        data.get('password'),
        data.get('role', 'User'),
        data.get('email'),
        data.get('phone'),
        data.get('age')
    )
    return jsonify({
        'success': success,
        'message': message
    }), 200 if success else 500



@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/x_ray')
def x_ray():
    return render_template('x_ray.html')


@app.route('/pres')
def pres():
    # Pass all available languages to the template
    return render_template('pres.html', languages=LANGUAGE_CODES)

@app.route('/diag_analysis')
def diag_analysis():
    return render_template('diag_analysis.html')

@app.route('/upload_diag', methods=['POST'])
def upload_diag():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Determine file type and process accordingly
        if file.filename.endswith('.pdf'):
            text = extract_text_from_pdf_diag(file)
            medical_values = extract_medical_values_diag(text)
            return render_template('diag_result.html', 
                                text=text,
                                medical_values=medical_values,
                                normal_ranges_diag=NORMAL_RANGES_diag)
        else:
            ocr_text, medicines = extract_text_from_image(file)
            analysis = analyze_handwritten_prescription(ocr_text)
            return render_template('diag_result.html',
                                text=ocr_text,
                                analysis=analysis,
                                medicines=medicines)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route('/pres_analyze', methods=['POST'])
@login_required
def pres_analyze():
    try:
        if 'prescription' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['prescription']
        target_language = request.form.get('language', 'English')
        
        # Get location from form data or fallback to IP-based
        lat = request.form.get('latitude')
        lon = request.form.get('longitude')
        
        if lat and lon:
            location = (float(lat), float(lon))
        else:
            location = get_current_location()
        
        # Read the file into memory
        file_stream = io.BytesIO(file.read())
        
        # Extract text and medicines from the image
        ocr_text, medicines = extract_text_from_image(file_stream)
        
        # Log extracted medicines for debugging
        logger.info(f"Initially extracted medicines: {medicines}")
        
        if not ocr_text:
            return jsonify({'error': 'No text could be extracted'}), 400
        
        # Analyze the prescription
        translate_to = LANGUAGE_CODES.get(target_language)
        analysis_result = analyze_handwritten_prescription(ocr_text, translate_to)
        
        # If no medicines were extracted, try to extract them from the analysis
        if not medicines:
            logger.info("No medicines from initial extraction, trying fallback...")
            # Use regex to find potential medicine patterns in the analysis
            med_pattern = r"(?i)(Tab|Tablet|Cap|Capsule|Inj|Injection|Rx|Syp|Syrup|Drp|Drops|Lotion|Cream|Gel|Amp|Ampoule|Vial|Medicine)\.?\s*([A-Za-z0-9\-\s]+(?:\d+\s*(?:mg|mcg|ml|g|IU|%|units))?)"
            medicines = re.findall(med_pattern, analysis_result)
            logger.info(f"Medicines from fallback extraction: {medicines}")
            
            # If still no medicines, generate a generic one from the OCR text
            if not medicines:
                # Create a generic "Medicine" entry with the first recognizable word
                words = re.findall(r'\b[A-Za-z]{4,}\b', ocr_text)
                if words:
                    potential_med = words[0].title()
                    medicines = [('Medicine', potential_med)]
                    logger.info(f"Created generic medicine entry: {medicines}")
        
        # Verify and correct medications
        verified_medicines = []
        if medicines:
            try:
                verified_medicines = final_medication_verification(medicines)
                logger.info(f"Verified medicines: {verified_medicines}")
            except Exception as e:
                logger.error(f"Error verifying medicines: {e}")
                verified_medicines = medicines  # Fallback to unverified medicines
        
        # Remove duplicate medications
        unique_medicines = []
        seen_med_names = set()
        for med_type, med_name in verified_medicines:
            if med_name.lower() not in seen_med_names:
                unique_medicines.append((med_type, med_name))
                seen_med_names.add(med_name.lower())
        
        # Ensure we have some medicine entries to show
        if not unique_medicines:
            # Add a placeholder medicine if none were found
            unique_medicines = [('Medicine', 'Unidentified Medication')]
            logger.info("Added placeholder medicine as none were found")
        
        # Get online pharmacy links for each medicine
        pharmacy_links = {}
        for _, med_name in unique_medicines:
            pharmacy_links[med_name] = search_online_pharmacies(med_name)
        
        # Predict condition based on medications
        condition_analysis = predict_disease_from_medications(verified_medicines)
        
        # Process pharmacy information
        pharmacies = []
        map_data = None
        
        if location:
            lat, lon = location
            try:
                # Get nearby pharmacies using Google Places API with OSM fallback
                search_radius = 5000  # 5 km
                pharmacies = get_nearby_pharmacies(lat, lon, radius=search_radius)
                
                # Logging for debugging
                logger.info(f"Found {len(pharmacies)} pharmacies near {lat}, {lon}")
                
                # Ensure all pharmacies have distance and other required fields
                for pharmacy in pharmacies:
                    # If pharmacy doesn't have distance, calculate it
                    if 'distance' not in pharmacy:
                        # Calculate distance using haversine formula
                        from math import radians, sin, cos, sqrt, atan2
                        
                        pharmacy_lat = pharmacy.get('latitude')
                        pharmacy_lon = pharmacy.get('longitude')
                        
                        if pharmacy_lat and pharmacy_lon:
                            R = 6371  # Earth radius in kilometers
                            dlat = radians(pharmacy_lat - lat)
                            dlon = radians(pharmacy_lon - lon)
                            
                            a = sin(dlat/2)**2 + cos(radians(lat)) * cos(radians(pharmacy_lat)) * sin(dlon/2)**2
                            c = 2 * atan2(sqrt(a), sqrt(1-a))
                            distance = round(R * c, 2)
                            pharmacy['distance'] = distance
                        else:
                            # Default distance if coordinates are missing
                            pharmacy['distance'] = 999.99
                    
                    # Ensure all required fields exist with defaults
                    if 'walking_time' not in pharmacy:
                        pharmacy['walking_time'] = round(pharmacy['distance'] * 12)
                    if 'driving_time' not in pharmacy:
                        pharmacy['driving_time'] = round(pharmacy['distance'] * 2)
                    if 'source' not in pharmacy:
                        pharmacy['source'] = 'Unknown'
                    if 'address' not in pharmacy or not pharmacy['address']:
                        pharmacy['address'] = 'Address unavailable'
                    
                    # Add any new Google-specific fields we want to display
                    if 'rating' in pharmacy:
                        # Format rating to one decimal place if present
                        pharmacy['rating_display'] = f"{pharmacy['rating']:.1f}/5" if pharmacy['rating'] else "No ratings"
                    
                    # Add a "map_url" for Google Maps directions
                    pharmacy['map_url'] = f"https://www.google.com/maps/dir/?api=1&destination={pharmacy['latitude']},{pharmacy['longitude']}"
                    
                    # Add an "open_status" display
                    if 'open_now' in pharmacy:
                        pharmacy['open_status'] = "Open now" if pharmacy['open_now'] else "Closed"
                
                # Sort pharmacies by distance to ensure the closest one is first
                pharmacies.sort(key=lambda x: x.get('distance', float('inf')))
                
                # Generate map data if pharmacies found
                if pharmacies:
                    azure_maps_key = AZURE_MAPS_KEY
                    
                    # Limit to 15 closest pharmacies for better map performance
                    map_pharmacies = pharmacies[:15] if len(pharmacies) > 15 else pharmacies
                    
                    map_data = {
                        'apiKey': azure_maps_key,
                        'center': {'latitude': lat, 'longitude': lon},
                        'pharmacies': map_pharmacies
                    }
                    
                    logger.info(f"Generated map data with {len(map_pharmacies)} pharmacies")
            except Exception as e:
                logger.error(f"Error getting pharmacy data: {e}")
                logger.exception(e)
        
        # Format medication data for easier use in the template
        formatted_medicines = []
        for med_type, med_name in unique_medicines:
            med_form = "Injectable" if med_type.lower() in ["inj", "injection", "amp", "ampoule", "vial"] else \
                       "Oral" if med_type.lower() in ["tab", "tablet", "cap", "capsule", "syp", "syrup"] else \
                       "Topical" if med_type.lower() in ["cream", "gel", "lotion", "ointment"] else "Other"
            
            formatted_medicines.append({
                "type": med_type,
                "name": med_name,
                "form": med_form,
                "pharmacy_links": pharmacy_links.get(med_name, {})
            })
        
        # Enhance the response data with pharmacy source info
        pharmacy_sources = {}
        if pharmacies:
            # Count pharmacies by source (Google vs OSM)
            for pharmacy in pharmacies:
                source = pharmacy.get('source', 'Unknown')
                pharmacy_sources[source] = pharmacy_sources.get(source, 0) + 1
        
        # Return complete result as JSON
        response_data = {
            'ocr_text': ocr_text,
            'analysis': analysis_result,
            'medicines': unique_medicines,  # Original format for compatibility
            'formatted_medicines': formatted_medicines,  # Enhanced format with links
            'condition_analysis': condition_analysis,
            'location': location,
            'pharmacies': pharmacies,
            'map_data': map_data,  # Provide structured map data instead of HTML
            # Add pharmacy summary for frontend display
            'pharmacy_summary': {
                'count': len(pharmacies),
                'closest': pharmacies[0].get('distance', 0) if pharmacies else None,
                'search_radius_km': search_radius / 1000,
                'sources': pharmacy_sources
            } if pharmacies else None
        }
        
        logger.info(f"Final medicines being sent to frontend: {unique_medicines}")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        logger.exception(e)
        return jsonify({'error': str(e)}), 500




@app.route('/analyze_x_ray', methods=['POST'])
@login_required
def analyze_x_ray():
    temp_path = None
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'})
        
        # Create temporary file with a unique name
        temp_fd, temp_path = tempfile.mkstemp()
        os.close(temp_fd)  # Close the file descriptor immediately
        
        # Save the uploaded file
        file.save(temp_path)
        
        # Get user info from session
        username = session.get('username', 'guest')
        user_id = session.get('user_id', 'guest_id')
        
        # Get form data
        image_type = request.form.get('image_type', 'X-ray')
        model_name = request.form.get('model_name', 'gemini-1.5-pro-vision')
        
        # Initialize analyzer
        analyzer = MedicalImageAnalyzer_x_ray(
            api_key=Gemini_API_KEY,
            gemini_model_name=model_name
        )
        
        # Perform analysis
        result = analyzer.analyze_medical_image(temp_path, image_type=image_type)
        
        # Save to Azure Data Lake Storage in the diagnosis subdirectory
        storage_success, storage_message = upload_file_tolake(
            username=username,
            user_id=user_id,
            file_path=temp_path,
            subdir="diagnosis",  # Save to diagnosis subdirectory
            file_name=f"{image_type}_{file.filename}"  # Prefix with image type
        )
        
        if storage_success:
            # Add storage information to the result
            result['storage'] = {
                'success': True,
                'message': f"File saved to your diagnosis folder in MediLocker",
                'path': storage_message
            }
        else:
            # Add storage error information to the result
            result['storage'] = {
                'success': False,
                'message': f"Analysis successful but file storage failed: {storage_message}"
            }
            logger.error(f"Failed to save x-ray to Azure: {storage_message}")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error analyzing x-ray: {str(e)}")
        return jsonify({'error': str(e)})
    
    finally:
        # Clean up the temporary file in the finally block
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as e:
                print(f"Error cleaning up temporary file: {e}")


@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    if request.method == 'POST':
        user_input = request.json.get('user_input')
        response = chat_function(user_input)
        return jsonify({'response': response})
    return render_template('chatbot.html')

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    data = request.json
    message = data.get('message')
    
    # Extract medical content from potentially mixed queries
    medical_content = extract_medical_content(message)
    
    # First check if the query is purely non-medical
    if is_non_medical_query(message) and not is_medical_query(medical_content):
        return jsonify({
            'response': get_non_medical_response(),
            'isDiseaseQuery': False
        })
    
    # Check if it's a disease query
    is_disease = is_disease_query(medical_content)
    
    if not is_disease:
        dataset_answer, dataset_source = search_all_datasets(message, medical_datasets)
        web_answer, web_source = search_medical_websites(message)
        response = get_enhanced_response(
            [{"role": "user", "content": message}],
            dataset_info=dataset_answer,
            web_info=web_answer
        )
        return jsonify({
            'response': response,
            'isDiseaseQuery': False
        })
    else:
        # For medical queries, include the first question right away in the response
        questions = get_default_symptom_questions(medical_content)
        
        # Add options to each question in the questions list
        for q in questions:
            q['options'] = get_disease_specific_options(q, medical_content)
            
        first_question = questions[0]["question"] if questions else "How severe are your symptoms?"
        options = questions[0]['options'] if questions else ["Mild", "Moderate", "Severe", "Very Severe"]
        
        # Include the first question directly in the response text
        response = f"I'll need to ask you some questions about your {medical_content}. First, {first_question}"
        
        return jsonify({
            'response': response,
            'isDiseaseQuery': True,
            'currentQuestion': {
                'question': first_question,
                'options': options,
                'questionIndex': 0,
                'type': questions[0]['type'] if questions else 'severity'
            },
            'medicalContent': medical_content,
            'allQuestions': questions  # Include all questions to reduce API calls
        })

# Add new endpoint to get options for a specific question type
@app.route('/get_question_options', methods=['GET'])
def get_question_options():
    disease = request.args.get('disease', '')
    question_type = request.args.get('questionType', '')
    question_text = request.args.get('question', '')
    
    # Create a question object to pass to the options function
    question_data = {
        "question": question_text,
        "type": question_type
    }
    
    # Get options specific to this question type
    options = get_disease_specific_options(question_data, disease)
    
    return jsonify({
        'options': options
    })

@app.route('/get_questions', methods=['GET'])
def get_questions_endpoint():
    disease = request.args.get('disease')
    questions = get_symptom_questions(disease)
    
    formatted_questions = []
    for q in questions:
        options = get_disease_specific_options(q, disease)
        formatted_questions.append({
            'question': q,
            'options': options
        })
    
    return jsonify({'questions': formatted_questions})

@app.route('/process_symptoms', methods=['POST'])
def process_symptoms_endpoint():
    data = request.json
    disease = data.get('disease')
    answers = data.get('answers')
    
    # Get clinical condition data
    clinical_conditions = search_clinical_conditions(disease)
    clinical_info = None
    if clinical_conditions:
        clinical_info = clinical_conditions[0]
    
    dataset_answer, dataset_source = search_all_datasets(disease, medical_datasets)
    web_answer, web_source = search_medical_websites(disease)
    
    response = format_medical_response(
        disease=disease,
        symptoms=answers,
        dataset_info=dataset_answer,
        web_info=web_answer
    )
    
    return jsonify({
        'response': response,
        'clinical_info': clinical_info
    })


@app.route('/index')
def serve_html():
    try:
        return render_template('index.html')
    except Exception as e:
        app.logger.error(f"Error rendering template: {str(e)}")
        abort(500)

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.static_folder, filename)

@app.errorhandler(500)
def server_error(error):
    return "Internal server error. Please check the application logs for details.", 500

def run_flask():
    print("Starting Flask server on port 5001...")
    try:
        app.run(debug=True, port=5001, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("Server stopped")

if __name__ == '__main__':
    import sys
    import subprocess
    if len(sys.argv) > 1 and sys.argv[1] == '--api':
        # Run as Flask API
        app.run(debug=True, port=5000)
    else:
        run_flask()

