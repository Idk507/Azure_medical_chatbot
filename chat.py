import os
import base64
import streamlit as st
try:
    from openai import AzureOpenAI
except ImportError:
    print("OpenAI package is not installed. Please install it using: pip install openai")
    # You could define a fallback implementation or placeholder
    class AzureOpenAI:
        def __init__(self, *args, **kwargs):
            print("WARNING: Using placeholder AzureOpenAI class")
        
        def __getattr__(self, name):
            def method(*args, **kwargs):
                return "OpenAI functionality not available. Please install the openai package."
            return method
import pandas as pd
import PyPDF2
# Update the import statement for docx
try:
    import docx  # Try importing first
except ImportError:
    print("Trying alternative docx import")
    try:
        from docx import Document  # python-docx package
    except ImportError:
        print("ERROR: Neither docx nor python-docx package is installed. Please run: pip install python-docx")
import glob
from pathlib import Path

# Add try-except for sklearn imports
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("ERROR: scikit-learn package is not installed. Please run: pip install scikit-learn")
    
    # Define minimal fallback implementations
    class TfidfVectorizer:
        def __init__(self, **kwargs):
            print("WARNING: Using placeholder TfidfVectorizer class")
        
        def fit_transform(self, texts):
            print("TfidfVectorizer functionality not available")
            # Return a simple representation that can be used minimally
            return [[1.0] * len(texts)]
        
        def transform(self, texts):
            print("TfidfVectorizer functionality not available")
            # Return a simple representation that can be used minimally
            return [[1.0] * len(texts)]
    
    def cosine_similarity(a, b):
        print("Cosine similarity functionality not available")
        # Return a dummy similarity matrix
        return [[0.5]]

import json
import jsonlines
import sqlite3
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from flask import Flask, request, jsonify
from flask_cors import CORS

# Azure OpenAI configuration
from dotenv import load_dotenv
load_dotenv()
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")

# Initialize Azure OpenAI Service client
client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=subscription_key,
    api_version="2024-05-01-preview"
)

# Initialize Document Analysis client
document_endpoint = os.getenv("DOCUMENT_ENDPOINT")
document_key = os.getenv("DOCUMENT_KEY")
document_client = DocumentAnalysisClient(
    endpoint=document_endpoint, 
    credential=AzureKeyCredential(document_key)
)

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

# Initialize Streamlit page with medical theme
st.set_page_config(
    page_title="Dr. OneMed - Virtual Medical Consultation",
    page_icon="👨‍⚕️",
    layout="wide"
)

# Customize the UI with medical theme
st.markdown("""
<style>
    .main {
        background-color: #f0f4f8;
    }
    .stTitle {
        color: #1a4f72;
    }
    .stTextInput {
        border: 2px solid #1a4f72;
    }
    .stAlert {
        background-color: #e3f2fd;
        border: 2px solid #1a4f72;
    }
</style>
""", unsafe_allow_html=True)

# Virtual Doctor welcome message
st.title("👨‍⚕️ Dr. OneMed - Virtual Consultation")
st.markdown("""
### Welcome to your virtual medical consultation

Before we begin:

**Consultation Guidelines:**
1. 🏥 Please describe your symptoms or concerns in detail
2. 📋 Mention any relevant medical history
3. 💊 Include information about current medications
4. ⏱️ Note symptom duration if applicable

**Important Medical Disclaimer:**
- This is a preliminary consultation tool
- Not a replacement for in-person medical care
- For emergencies, call your local emergency services immediately
- Always follow up with your healthcare provider

""")

# Add constant for dataset path
DATASET_PATH = r"C:\Users\jeffr\OneDrive\Desktop\hackacloud\dataset"

# Add dataset categories and file mappings
DATASET_CATEGORIES = {
    'blood_tests': ['Blood_Test_Report_Sample'],
    'diabetes': ['Diabetes Classification', 'diabetes'],
    'symptoms': ['Diseases_Symptoms', 'symptom_Description', 'symptom_precaution', 'Symptom_severity',
                'symptom-disease-test-dataset', 'symptom-disease-train-dataset'],
    'reports': ['structured_report_data'],
    'tests': ['urine'],
    'medications': ['Medicine_Details']
}

# Initialize datasets
def load_medical_datasets():
    datasets = {}
    
    # Process CSV files
    csv_files = glob.glob(os.path.join(DATASET_PATH, "*.csv"))
    for file_path in csv_files:
        try:
            name = os.path.splitext(os.path.basename(file_path))[0]
            df = pd.read_csv(file_path)
            df['source'] = name
            datasets[name] = df
        except Exception as e:
            st.warning(f"Error reading CSV {file_path}: {str(e)}")
    
    # Process JSON files
    json_files = glob.glob(os.path.join(DATASET_PATH, "*.json"))
    for file_path in json_files:
        try:
            name = os.path.splitext(os.path.basename(file_path))[0]
            with open(file_path, 'r') as f:
                data = json.load(f)
            df = pd.json_normalize(data)
            df['source'] = name
            datasets[name] = df
        except Exception as e:
            st.warning(f"Error reading JSON {file_path}: {str(e)}")
    
    # Process JSONL files with improved error handling
    jsonl_files = glob.glob(os.path.join(DATASET_PATH, "*.jsonl"))
    for file_path in jsonl_files:
        try:
            name = os.path.splitext(os.path.basename(file_path))[0]
            
            # Special handling for Diseases_Symptoms.jsonl
            if name == "Diseases_Symptoms":
                try:
                    df = pd.read_csv(file_path.replace('.jsonl', '.csv'))
                    df['source'] = name
                    datasets[name] = df
                    continue
                except Exception as csv_error:
                    st.warning(f"Error reading CSV fallback for {name}: {str(csv_error)}")
            
            # Try reading as CSV first
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
                df['source'] = name
                datasets[name] = df
                continue
            except Exception as csv_error:
                pass
            
            # Try reading as JSONL with multiple encodings
            for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
                try:
                    data = []
                    with open(file_path, 'r', encoding=encoding) as f:
                        for line in f:
                            line = line.strip()
                            if not line:  # Skip empty lines
                                continue
                            try:
                                # Try parsing as regular JSON
                                obj = json.loads(line)
                                data.append(obj)
                            except json.JSONDecodeError:
                                # Try converting CSV-like line to JSON
                                if ',' in line:
                                    values = line.split(',')
                                    if len(values) > 1:
                                        obj = {"symptom": values[0], "disease": values[1]}
                                        data.append(obj)
                    
                    if data:  # Only create dataframe if we have data
                        df = pd.json_normalize(data)
                        df['source'] = name
                        datasets[name] = df
                        break  # Break the encoding loop if successful
                    
                except Exception as e:
                    continue  # Try next encoding
                    
        except Exception as e:
            st.warning(f"Error reading JSONL {file_path}: {str(e)}")
    
    # Process SQLite database
    db_path = os.path.join(DATASET_PATH, "database.db")
    if (os.path.exists(db_path)):
        try:
            conn = sqlite3.connect(db_path)
            tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
            for table_name in tables['name']:
                df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                df['source'] = f"database_{table_name}"
                datasets[f"database_{table_name}"] = df
            conn.close()
        except Exception as e:
            st.warning(f"Error reading database: {str(e)}")
    
    return datasets

# Initialize TF-IDF vectorizer
@st.cache_resource
def get_vectorizer():
    return TfidfVectorizer(stop_words='english')

def search_all_datasets(query, datasets, threshold=0.3):
    if not datasets:
        return None, None
    
    vectorizer = get_vectorizer()
    best_match = None
    best_score = threshold
    source = None
    
    for dataset_name, df in datasets.items():
        try:
            # Convert all values to string before joining
            df = df.astype(str)
            
            # Get available columns
            columns = df.columns.str.lower()
            
            # Determine dataset type and format response accordingly
            if 'drug_name' in columns or 'medicine_name' in columns or 'drug' in columns:
                drug_col = next(col for col in df.columns if col.lower() in ['drug_name', 'medicine_name', 'drug'])
                comp_col = next((col for col in df.columns if 'composition' in col.lower()), None)
                uses_col = next((col for col in df.columns if 'uses' in col.lower() or 'indication' in col.lower()), None)
                
                search_text = df[drug_col]
                if comp_col:
                    search_text += " " + df[comp_col]
                if uses_col:
                    search_text += " " + df[uses_col]
                    
            elif 'disease' in columns or 'condition' in columns:
                disease_col = next(col for col in df.columns if col.lower() in ['disease', 'condition'])
                desc_col = next((col for col in df.columns if 'description' in col.lower()), None)
                
                search_text = df[disease_col]
                if desc_col:
                    search_text += " " + df[desc_col]
                    
            elif 'symptom' in columns:
                search_text = df['symptom']
                if 'disease' in columns:
                    search_text += " " + df['disease']
            else:
                search_text = df.apply(lambda x: ' '.join(x.values), axis=1)
            
            vectors = vectorizer.fit_transform(search_text)
            query_vector = vectorizer.transform([query])
            
            similarities = cosine_similarity(query_vector, vectors)[0]
            best_match_idx = similarities.argmax()
            
            if similarities[best_match_idx] > best_score:
                best_score = similarities[best_match_idx]
                row = df.iloc[best_match_idx]
                
                # Format response based on available columns
                if 'drug_name' in columns or 'medicine_name' in columns or 'drug' in columns:
                    drug_col = next(col for col in df.columns if col.lower() in ['drug_name', 'medicine_name', 'drug'])
                    response_parts = [f"Medicine: {row[drug_col]}"]
                    
                    for col in df.columns:
                        if col.lower() not in ['source', drug_col.lower()] and str(row[col]) != 'nan':
                            response_parts.append(f"{col}: {row[col]}")
                    
                    best_match = "\n".join(response_parts)
                    
                elif 'disease' in columns or 'condition' in columns:
                    disease_col = next(col for col in df.columns if col.lower() in ['disease', 'condition'])
                    response_parts = [f"Condition: {row[disease_col]}"]
                    
                    for col in df.columns:
                        if col.lower() not in ['source', disease_col.lower()] and str(row[col]) != 'nan':
                            response_parts.append(f"{col}: {row[col]}")
                    
                    best_match = "\n".join(response_parts)
                    
                else:
                    best_match = '\n'.join(f"{col}: {val}" for col, val in row.items() 
                                         if col != 'source' and str(val).lower() != 'nan')
                
                source = dataset_name
        
        except Exception as e:
            st.error(f"Error searching {dataset_name} dataset: {str(e)}")
            continue
    
    return best_match, source

def analyze_document(file_path):
    """Analyze document using Azure Document Intelligence"""
    try:
        with open(file_path, "rb") as f:
            poller = document_client.begin_analyze_document(
                "prebuilt-document", document=f
            )
            result = poller.result()
            
            # Extract text content
            content = " ".join([p.content for p in result.paragraphs])
            
            # Extract key phrases and entities
            entities = [entity.content for entity in result.entities]
            key_phrases = [kp.content for kp in result.key_phrases]
            
            return {
                'content': content,
                'entities': entities,
                'key_phrases': key_phrases
            }
    except Exception as e:
        st.warning(f"Error analyzing document {file_path}: {str(e)}")
        return None

def load_unstructured_data():
    documents = []
    
    # Process all files in the dataset directory
    for ext in ['*.txt', '*.pdf', '*.docx']:
        files = glob.glob(os.path.join(DATASET_PATH, "**", ext), recursive=True)
        for file_path in files:
            try:
                # Use Document Intelligence for PDFs and complex documents
                if (file_path.endswith(('.pdf', '.docx'))):
                    doc_analysis = analyze_document(file_path)
                    if doc_analysis:
                        content = doc_analysis['content']
                        if doc_analysis['entities'] or doc_analysis['key_phrases']:
                            content += "\nKey Information: " + ", ".join(
                                set(doc_analysis['entities'] + doc_analysis['key_phrases'])
                            )
                else:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                
                documents.append({
                    'content': content,
                    'source': os.path.relpath(file_path, DATASET_PATH),
                    'type': os.path.splitext(file_path)[1][1:]
                })
            except Exception as e:
                st.warning(f"Error reading {file_path}: {str(e)}")
    
    return documents

def search_unstructured_data(query, documents, threshold=0.3):
    if not documents:
        return None, None
    
    vectorizer = get_vectorizer()
    
    try:
        # Create corpus from documents
        corpus = [doc['content'] for doc in documents]
        vectors = vectorizer.fit_transform(corpus)
        query_vector = vectorizer.transform([query])
        
        # Calculate similarity scores
        similarities = cosine_similarity(query_vector, vectors)[0]
        best_match_idx = similarities.argmax()
        
        if similarities[best_match_idx] >= threshold:
            # Extract relevant context around the best match
            content = documents[best_match_idx]['content']
            source = f"{documents[best_match_idx]['type']}: {documents[best_match_idx]['source']}"
            
            # Extract relevant snippet (simple approach)
            sentences = content.split('.')
            for i, sentence in enumerate(sentences):
                if query.lower() in sentence.lower():
                    start = max(0, i-1)
                    end = min(len(sentences), i+2)
                    context = '. '.join(sentences[start:end])
                    return context, source
            
            # If no direct match, return a portion of the best matching document
            return content[:500] + "...", source
            
        return None, None
        
    except Exception as e:
        st.error(f"Error searching unstructured data: {str(e)}")
        return None, None

# Add medical website search configurations
MEDICAL_WEBSITES = [
    {
        'name': 'Mayo Clinic',
        'search_url': 'https://www.mayoclinic.org/search/search-results?q={}',
        'base_url': 'https://www.mayoclinic.org'
    },
    {
        'name': 'WebMD',
        'search_url': 'https://www.webmd.com/search/search_results/default.aspx?query={}',
        'base_url': 'https://www.webmd.com'
    },
    {
        'name': 'Healthline',
        'search_url': 'https://www.healthline.com/search?q1={}',
        'base_url': 'https://www.healthline.com'
    },
    {
        'name': 'Penn Medicine',
        'search_url': 'https://www.pennmedicine.org/search-results?q={}',
        'base_url': 'https://www.pennmedicine.org'
    },
    {
        'name': 'Medical News Today',
        'search_url': 'https://www.medicalnewstoday.com/search?q={}',
        'base_url': 'https://www.medicalnewstoday.com'
    },
    {
        'name': 'Health Direct',
        'search_url': 'https://www.healthdirect.gov.au/search-results?q={}',
        'base_url': 'https://www.healthdirect.gov.au'
    },
    {
        'name': 'Lung.org',
        'search_url': 'https://www.lung.org/search?query={}',
        'base_url': 'https://www.lung.org'
    }
]

def search_medical_websites(query):
    results = []
    
    for site in MEDICAL_WEBSITES:
        try:
            search_url = site['search_url'].format(quote_plus(query))
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
            }
            response = requests.get(search_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Generic search for article titles and links
                article_elements = soup.find_all(['article', 'div'], class_=lambda x: x and any(term in x.lower() for term in ['result', 'article', 'item']))
                
                for element in article_elements[:2]:
                    title_elem = element.find(['h2', 'h3', 'a'])
                    if title_elem:
                        title = title_elem.get_text().strip()
                        url = title_elem.get('href', '')
                        if url and not url.startswith('http'):
                            url = site['base_url'] + url
                        if title and url:
                            results.append(f"[{site['name']}] {title}\n{url}")
                
        except Exception as e:
            st.warning(f"Error searching {site['name']}: {str(e)}")
            continue
    
    if results:
        return "\n\n".join(results), "Medical Websites"
    return None, None

def get_enhanced_response(messages, dataset_info=None, web_info=None):
    """Get enhanced response with doctor-like formatting"""
    try:
        system_prompt = {
            "role": "system",
            "content": """As Dr. OneMed, structure your responses as follows:
            
            1. Initial Assessment:
               - Brief acknowledgment of the concern
               - Clarifying questions if needed
            
            2. Medical Discussion:
               - Clinical explanation
               - Relevant medical terms with definitions
               - Possible causes or factors
            
            3. Recommendations:
               - Specific medical guidance
               - Lifestyle or preventive measures
               - Medication considerations (if applicable)
            
            4. Professional Guidance:
               - Follow-up recommendations
               - Red flags to watch for
               - When to seek immediate care
            
            Maintain a professional, caring tone throughout."""
        }
        
        context_message = {
            "role": "system",
            "content": f"Medical context: {dataset_info}" if dataset_info else ""
        }
        
        web_context = {
            "role": "system",
            "content": f"Additional information: {web_info}" if web_info else ""
        }
        
        enhanced_messages = [system_prompt, context_message, web_context] + messages[-5:]
        
        response = client.chat.completions.create(
            model=deployment,
            messages=enhanced_messages,
            max_tokens=800,
            temperature=0.7,
            presence_penalty=0.6,
            frequency_penalty=0.2,
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        st.error(f"Error getting enhanced response: {str(e)}")
        return "I apologize, but I'm having trouble generating a response. Please try rephrasing your question."

def is_medical_query(query):
    """Check if the query is related to medical topics"""
    medical_keywords = [
        # Disease keywords
        'disease', 'condition', 'syndrome', 'disorder', 'infection', 'illness',
        'fever', 'pain', 'ache', 'inflammation', 'itis', 'flu', 'virus',
        'infection', 'diabetes', 'cancer', 'arthritis',
        
        # Symptom keywords
        'symptom', 'cough', 'headache', 'nausea', 'vomiting', 'dizziness',
        'fatigue', 'weakness', 'shortness of breath', 'blood pressure',
        
        # Medical procedure keywords
        'surgery', 'treatment', 'therapy', 'medication', 'prescription',
        'diagnosis', 'prognosis', 'doctor', 'hospital', 'clinic', 'vaccine',
        
        # Body parts
        'heart', 'lung', 'liver', 'kidney', 'brain', 'stomach', 'intestine',
        'joint', 'muscle', 'bone', 'skin', 'blood', 
        
        # Healthcare terms
        'health', 'medical', 'patient', 'nurse', 'physician', 'healthcare'
    ]
    
    # Add more specific symptom patterns
    symptom_patterns = [
        'i have', 'i feel', 'i am experiencing', 'suffering from', 
        'symptoms of', 'pain in', 'hurts when'
    ]
    
    query_lower = query.lower()
    has_medical_keywords = any(keyword in query_lower for keyword in medical_keywords)
    has_symptom_pattern = any(pattern in query_lower for pattern in symptom_patterns)
    
    return has_medical_keywords or has_symptom_pattern

# Add alias function for backward compatibility with app.py
def is_disease_query(query):
    """Alias for is_medical_query to maintain backward compatibility"""
    return is_medical_query(query)

def is_non_medical_query(query):
    """Check if the query is clearly non-medical (programming, etc.)"""
    non_medical_keywords = [
        # Programming terms
        'code', 'programming', 'python', 'javascript', 'java', 'c++', 'function',
        'algorithm', 'variable', 'compiler', 'runtime', 'database', 'array',
        'class', 'object', 'method', 'import', 'framework', 'api', 'library',
        'syntax', 'script', 'debugging', 'html', 'css', 'json', 'xml',
        
        # Other non-medical domains
        'finance', 'stock', 'investment', 'recipe', 'cooking', 'travel',
        'sports', 'game', 'movie', 'politics', 'weather', 'news'
    ]
    
    # Code patterns that strongly indicate programming queries
    code_patterns = [
        'def ', 'import ', 'class ', 'print(', '```python', 'for ', 'while ',
        'if __name__', '.py', 'return ', '#include', 'public static void',
        'function(', 'var ', 'const ', 'let ', '```java', '```js', '```c++'
    ]
    
    query_lower = query.lower()
    
    # Don't flag as non-medical if it contains symptom description patterns
    symptom_patterns = ['i have', 'i feel', 'i am experiencing', 'suffering from']
    if any(pattern in query_lower for pattern in symptom_patterns):
        return False
    
    # Check for code block markers
    if '```' in query:
        return True
    
    # Check for explicit programming keywords
    if any(keyword in query_lower for keyword in non_medical_keywords):
        # If it also contains medical terms, it might be a medical informatics question
        if is_medical_query(query):
            return False
        return True
        
    # Check for code patterns that strongly indicate programming
    if any(pattern in query for pattern in code_patterns):
        return True
        
    return False

def extract_medical_content(query):
    """Extract medical-related content from a mixed query"""
    # If the query doesn't appear to be mixed, return it unchanged
    if not (is_medical_query(query) and is_non_medical_query(query)):
        return query
    
    # List of medical content introduction patterns
    medical_patterns = [
        "i have", "i feel", "i am experiencing", "suffering from",
        "symptoms of", "pain in", "hurts when", "diagnosed with"
    ]
    
    # Try to extract the medical portion
    query_lower = query.lower()
    for pattern in medical_patterns:
        if pattern in query_lower:
            # Get the part of the string starting from the pattern
            pattern_index = query_lower.find(pattern)
            medical_part = query[pattern_index:]
            
            # Find the end of the medical part (if followed by programming terms)
            end_markers = ["code", "python", "program", "function", "```"]
            end_index = len(medical_part)
            for marker in end_markers:
                marker_index = medical_part.lower().find(marker)
                if marker_index > 0 and marker_index < end_index:
                    end_index = marker_index
            
            medical_part = medical_part[:end_index].strip()
            if len(medical_part) > 10:  # Make sure we have a meaningful chunk
                return medical_part
    
    # If no clear medical part found, just return the original
    return query

def get_non_medical_response():
    """Return a standard response for non-medical queries"""
    return "I'm Dr. OneMed, a virtual medical consultant designed to answer health and medical questions only. I cannot provide assistance with programming, financial advice, or other non-medical topics. Please feel free to ask me any medical or health-related questions you may have."

def get_symptom_questions(disease):
    """Generate structured symptom questions dynamically based on disease-related symptoms."""
    try:
        def get_disease_symptoms(disease):
            """Fetch symptoms associated with a disease using Clinical Tables API"""
            try:
                params = {
                    'terms': disease,
                    'df': 'term_icd9_code,primary_name,symptoms'
                }
                response = requests.get(CLINICAL_TABLES_API, params=params)
                if response.status_code == 200:
                    data = response.json()
                    if len(data) == 4 and data[3]:  # API returns [data_count, ids, display_terms, [conditions]]
                        # Parse symptoms from the response
                        # Since Clinical Tables API doesn't directly provide symptoms,
                        # we'll use common symptom associations
                        common_symptoms = {
                            'gastroenteritis': ['nausea', 'vomiting', 'diarrhea', 'abdominal pain'],
                            'flu': ['fever', 'cough', 'fatigue', 'body aches'],
                            'migraine': ['headache', 'nausea', 'sensitivity to light', 'vision changes'],
                            'asthma': ['wheezing', 'shortness of breath', 'chest tightness', 'coughing']
                        }
                        
                        # Get the condition name from the API response
                        condition = data[3][0][1].lower() if data[3][0] else disease.lower()
                        return common_symptoms.get(condition, ['fever', 'pain', 'fatigue', 'discomfort'])
                        
                # Fallback: Return generic symptoms
                return ['fever', 'pain', 'fatigue', 'discomfort']
                
            except Exception as e:
                st.warning(f"Error fetching symptoms from Clinical Tables API: {str(e)}")
                # Fallback symptoms if API fails
                return ['fever', 'pain', 'fatigue', 'discomfort']
        
        symptoms = get_disease_symptoms(disease)
        
        prompt = f"""As a medical professional, generate key assessment questions for {disease}.
        Focus on these symptoms: {', '.join(symptoms)}.
        Structure the response as a Python list of dictionaries with 'question' and 'type' keys.
        Question types should be one of: 'severity', 'duration', 'frequency', 'symptom', 'medication', 'history'.
        Example format:
        [
            {{"question": "How severe is your {symptoms[0]}?", "type": "severity"}},
            {{"question": "How long have you had {symptoms[0]}?", "type": "duration"}}
        ]
        Include 4-6 most relevant questions for {disease} assessment."""
        
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Replace with your model deployment
            messages=[
                {"role": "system", "content": "You are a medical professional creating a symptom assessment questionnaire."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.3
        )
        
        # Parse response
        content = response["choices"][0]["message"]["content"]
        if '[' in content and ']' in content:
            questions = eval(content[content.find('['):content.find(']') + 1])
            return questions
        
        # Default questions if parsing fails
        return [
            {"question": f"How long have you experienced {symptoms[0]}?", "type": "duration"} if symptoms else {"question": "How long have you experienced these symptoms?", "type": "duration"},
            {"question": "How severe are your symptoms?", "type": "severity"},
            {"question": "Are you taking any medications?", "type": "medication"},
            {"question": "Have you had similar symptoms before?", "type": "history"}
        ]
        
    except Exception as e:
        st.error(f"Error generating questions: {str(e)}")
        return [
            {"question": "Duration of symptoms?", "type": "duration"},
            {"question": "Severity level?", "type": "severity"},
            {"question": "Current medications?", "type": "medication"}
        ]

def get_disease_specific_options(question_data, disease):
    """Get dynamic options based on the question type and disease"""
    question = question_data["question"]
    q_type = question_data["type"]
    
    try:
        # First check Clinical Tables API for disease-specific information
        clinical_conditions = search_clinical_conditions(disease)
        
        if q_type == "symptom" and clinical_conditions:
            # Use the first matching condition as reference
            condition = clinical_conditions[0]
            return ["Select answer", f"Confirmed {condition['name']}", 
                   f"Suspected {condition['name']}", "Different condition", 
                   "Not sure"]
        
        # Base options based on type
        if q_type == "severity":
            return ["Select severity", "Mild", "Moderate", "Severe", "Very Severe"]
        elif q_type == "duration":
            return ["Select duration", "Less than 24 hours", "1-3 days", "3-7 days", "1-2 weeks", "More than 2 weeks"]
        elif q_type == "frequency":
            return ["Select frequency", "Rarely", "Occasionally", "Frequently", "Constantly"]
        elif q_type == "medication":
            return ["Select option", "No medications", "Over-the-counter only", "Prescription medications", "Both OTC and prescription"]
        elif q_type == "history":
            return ["Select answer", "Never before", "Once before", "Multiple times", "Chronic condition"]
        elif q_type == "symptom":
            # Generate symptom-specific options based on disease
            symptom_options = get_common_symptoms(disease)
            return ["Select answer"] + symptom_options
        else:
            return ["Select option", "Yes", "No", "Not sure"]
            
    except Exception as e:
        st.error(f"Error generating options: {str(e)}")
        return ["Select option", "Yes", "No", "Not sure"]

def format_medical_response(disease, symptoms, dataset_info, web_info):
    """Format response like a human medical conversation"""
    try:
        # Check Clinical Tables API for disease information
        clinical_conditions = search_clinical_conditions(disease)
        clinical_info = ""
        if clinical_conditions:
            condition = clinical_conditions[0]
            clinical_info = f"\nClinical Classification: {condition['name']} (ICD-9: {condition['icd9_code']})"
        
        # Convert symptoms dict to formatted string
        symptom_text = "\n".join([f"- {q}: {a}" for q, a in symptoms.items()])
        
        prompt = f"""As Dr. OneMed, review the patient's symptoms for {disease}:

        Patient Assessment:
        {symptom_text}
        {clinical_info}

        Provide a warm, conversational response that includes:
        1. Acknowledgment of their symptoms
        2. Clear explanation of possible causes
        3. Practical recommendations
        4. Next steps and when to seek immediate care

        Use this context (but don't mention it directly):
        {dataset_info if dataset_info else ''}
        {web_info if web_info else ''}

        Keep the tone friendly and professional, like talking to a patient in person."""

        response = client.chat.completions.create(
            model=deployment,
            messages=[{
                "role": "system",
                "content": """You are Dr. OneMed, a warm and professional doctor who explains things clearly.
                Always maintain a caring, conversational tone while being thorough and professional.
                Use simple language first, then explain medical terms if needed.
                Make the patient feel heard and understood."""
            }, {
                "role": "user",
                "content": prompt
            }],
            max_tokens=1000,
            temperature=0.7,
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"I hear your concerns about {disease}. However, I'm having trouble providing a complete response right now. Please let me know if you'd like to try again, or consider consulting with a healthcare provider for immediate assistance."

def get_symptom_options():
    """Get a list of symptom options from the dataset"""
    try:
        symptom_dataset = medical_datasets.get('Diseases_Symptoms')
        if symptom_dataset is not None and 'symptom' in symptom_dataset.columns:
            symptoms = symptom_dataset['symptom'].unique().tolist()
            return symptoms
        else:
            return ["Fever", "Cough", "Fatigue", "Headache", "Shortness of breath"]
    except Exception as e:
        st.error(f"Error getting symptom options: {str(e)}")
        return ["Fever", "Cough", "Fatigue", "Headache", "Shortness of breath"]

def get_common_symptoms(disease):
    """Get common symptoms for a specific disease using OpenAI"""
    try:
        prompt = f"""As a medical professional, list 4-6 common symptoms for {disease}.
        Format as a simple Python list of strings.
        Example: ["Fever", "Cough", "Fatigue", "Headache"]
        Focus on the most typical symptoms associated with {disease}."""
        
        response = client.chat.completions.create(
            model=deployment,
            messages=[{
                "role": "system",
                "content": "You are a medical professional providing symptom lists."
            }, {
                "role": "user",
                "content": prompt
            }],
            max_tokens=100,
            temperature=0.3
        )
        
        content = response.choices[0].message.content
        if '[' in content and ']' in content:
            list_str = content[content.find('['):content.find(']') + 1]
            symptoms = eval(list_str)
            return symptoms
        else:
            return ["Symptom 1", "Symptom 2", "Symptom 3", "Symptom 4"]
            
    except Exception as e:
        st.error(f"Error getting symptoms: {str(e)}")
        return ["Fever", "Cough", "Fatigue", "Headache"]

# Add Clinical Tables API configuration
CLINICAL_TABLES_API = "https://clinicaltables.nlm.nih.gov/api/conditions/v3/search"

def search_clinical_conditions(query):
    """Search conditions using Clinical Tables API"""
    try:
        params = {
            'terms': query,
            'df': 'term_icd9_code,primary_name'
        }
        response = requests.get(CLINICAL_TABLES_API, params=params)
        if response.status_code == 200:
            data = response.json()
            if len(data) == 4:  # API returns [data_count, ids, display_terms, [conditions]]
                conditions = data[3]
                formatted_conditions = []
                for condition in conditions:
                    if condition[0]:  # ICD9 code
                        formatted_conditions.append({
                            'icd9_code': condition[0],
                            'name': condition[1]
                        })
                return formatted_conditions
        return []
    except Exception as e:
        st.warning(f"Error searching clinical conditions: {str(e)}")
        return []

# Load all datasets
medical_datasets = load_medical_datasets()

# Load unstructured documents
unstructured_documents = load_unstructured_data()

# Initialize session state for chat history
if "messages" not in st.session_state:
    initial_message = "Hi, I'm Dr. OneMed, your AI medical consultant here to assist you. How can I help you today?"
    st.session_state.messages = [
        {
            "role": "system",
            "content": """You are Dr. OneMed, an AI medical consultant. Respond like a professional doctor would:
            1. Start with a brief, warm professional greeting
            2. Ask clarifying questions when needed
            3. Provide structured medical explanations
            4. Give clear, specific recommendations
            5. Use professional medical terminology with explanations
            6. Always maintain a caring but professional tone
            7. End with appropriate medical guidance and follow-up recommendations
            Remember to emphasize the importance of consulting with a healthcare provider."""
        },
        {
            "role": "assistant",
            "content": initial_message
        }
    ]
    st.session_state.collecting_symptoms = False
    st.session_state.current_disease = None
    st.session_state.symptoms = []
    st.session_state.available_symptoms = get_symptom_options()
    st.session_state.symptom_index = 0  # Initialize symptom index
    st.session_state.symptom_questions = []  # Initialize symptom questions
    st.session_state.show_dropdowns = False
    st.session_state.responses = {}
    st.session_state.submitted = False
    st.session_state.current_question = 0  # Track current question
    st.session_state.answers = {}  # Store answers
    st.session_state.form_key = 0  # Add form key for unique form IDs
    st.session_state.review_mode = False  # Add review mode flag

# Display chat history
for message in st.session_state.messages[1:]:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    elif message["role"] == "assistant":
        st.chat_message("assistant").write(message["content"])

# Handle user input
user_input = st.chat_input("Ask your medical question...")
if user_input:
    # Display user's message first
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    try:
        # Extract medical content from potentially mixed queries
        medical_content = extract_medical_content(user_input)
        
        # First check if query is purely non-medical
        if is_non_medical_query(user_input) and not is_medical_query(medical_content):
            non_medical_response = get_non_medical_response()
            st.chat_message("assistant").write(non_medical_response)
            st.session_state.messages.append({"role": "assistant", "content": non_medical_response})
        elif is_medical_query(medical_content) and not st.session_state.collecting_symptoms:
            # Initialize symptom collection
            st.session_state.collecting_symptoms = True
            st.session_state.current_disease = medical_content
            st.session_state.symptom_questions = get_symptom_questions(medical_content)
            st.session_state.symptoms = []
            st.session_state.show_dropdowns = True
            st.session_state.submitted = False
            st.session_state.responses = {}
            st.session_state.current_question = 0  # Reset to first question

            # First acknowledge the user's query and display first question immediately
            initial_response = f"Thank you for consulting about your health concern. I'll need to ask you a few questions to better understand your condition."
            st.chat_message("assistant").write(initial_response)
            st.session_state.messages.append({"role": "assistant", "content": initial_response})
            
            # Force a rerun to display the symptom collection UI immediately
            st.rerun()
        
        elif st.session_state.collecting_symptoms:
            pass  # Skip this branch as we're handling symptoms with dropdowns
            
        else:
            # Handle non-disease queries as before
            st.session_state.messages.append({"role": "user", "content": user_input})
    
            try:
                # Collect context from all sources
                dataset_answer, dataset_source = search_all_datasets(user_input, medical_datasets)
                web_answer, web_source = search_medical_websites(user_input)
                local_answer, local_source = search_unstructured_data(user_input)
                
                # Combine all available context
                context = []
                if dataset_answer:
                    context.append(f"Dataset ({dataset_source}): {dataset_answer}")
                if web_answer:
                    context.append(f"Web Sources: {web_answer}")
                if local_answer:
                    context.append(f"Local Source ({local_source}): {local_answer}")
                
                # Get enhanced response with context
                enhanced_response = get_enhanced_response(
                    st.session_state.messages,
                    dataset_info=dataset_answer,
                    web_info=web_answer
                )
                
                # Format final response - REMOVE SOURCES
                assistant_reply = enhanced_response
                
                # Display assistant response
                st.chat_message("assistant").write(assistant_reply)
                st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
            
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Add symptom collection UI outside the user input handler
if st.session_state.show_dropdowns and not st.session_state.submitted:
    with st.container():
        st.write("### Symptom Assessment")
        
        if not st.session_state.review_mode:
            # Display the current question
            if st.session_state.current_question < len(st.session_state.symptom_questions):
                question = st.session_state.symptom_questions[st.session_state.current_question]
                options = get_disease_specific_options(question, st.session_state.current_disease)
                
                st.write(f"Question {st.session_state.current_question + 1} of {len(st.session_state.symptom_questions)}")
                
                # Create form for the current question
                with st.form(key=f"question_form_{st.session_state.form_key}"):
                    answer = st.selectbox(
                        label=question["question"],
                        options=options,
                        key=f"symptom_{st.session_state.current_question}"
                    )
                    
                    cols = st.columns([1, 1, 1])
                    with cols[0]:
                        back = st.form_submit_button("← Back") if st.session_state.current_question > 0 else None
                    with cols[2]:
                        next_button = st.form_submit_button("Next →" if st.session_state.current_question < len(st.session_state.symptom_questions) - 1 else "Review")
                    
                    if next_button:
                        if not answer.startswith("Select"):
                            st.session_state.answers[question["question"]] = answer
                            st.session_state.current_question += 1
                            st.session_state.form_key += 1
                            if st.session_state.current_question >= len(st.session_state.symptom_questions):
                                st.session_state.review_mode = True
                            st.rerun()
                        else:
                            st.error("Please select an option before proceeding.")
                    
                    if back and st.session_state.current_question > 0:
                        st.session_state.current_question -= 1
                        st.session_state.form_key += 1
                        st.rerun()
        
        else:
            # Review mode
            st.write("Please review your answers:")
            for i, (q, a) in enumerate(st.session_state.answers.items()):
                st.write(f"**Q{i+1}: {q}**")
                st.write(f"Answer: {a}")
            
            cols = st.columns([1, 1, 1])
            with cols[0]:
                if st.button("← Edit Answers"):
                    st.session_state.review_mode = False
                    st.session_state.form_key += 1
                    st.rerun()
            with cols[2]:
                if st.button("Submit →"):
                    # Process answers and generate response
                    dataset_answer, dataset_source = search_all_datasets(st.session_state.current_disease, medical_datasets)
                    web_answer, web_source = search_medical_websites(st.session_state.current_disease)
                    
                    assistant_reply = format_medical_response(
                        st.session_state.current_disease,
                        st.session_state.answers,
                        dataset_answer,
                        web_answer
                    )
                    
                    # Display the response
                    st.chat_message("assistant").write(assistant_reply)
                    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
                    
                    # Reset all states
                    st.session_state.current_disease = None
                    st.session_state.collecting_symptoms = False
                    st.session_state.show_dropdowns = False
                    st.session_state.submitted = True
                    st.session_state.current_question = 0
                    st.session_state.answers = {}
                    st.session_state.review_mode = False
                    st.rerun()

def get_default_symptom_questions(symptom):
    """Get default questions for common symptoms without making API calls"""
    default_questions = {
        "fever": [
            {"question": "How high is your fever?", "type": "severity"},
            {"question": "How long have you had the fever?", "type": "duration"},
            {"question": "Are you experiencing any other symptoms along with fever?", "type": "symptom"},
            {"question": "Have you taken any medication for the fever?", "type": "medication"},
            {"question": "Have you had similar fevers before?", "type": "history"}
        ],
        "headache": [
            {"question": "How would you rate your headache pain?", "type": "severity"},
            {"question": "How long have you had this headache?", "type": "duration"},
            {"question": "Where is the pain located in your head?", "type": "symptom"},
            {"question": "Does anything make your headache better or worse?", "type": "symptom"},
            {"question": "Have you taken any medication for it?", "type": "medication"}
        ],
        "cough": [
            {"question": "Is your cough dry or productive (with phlegm)?", "type": "symptom"},
            {"question": "How long have you been coughing?", "type": "duration"},
            {"question": "Do you cough more at any particular time of day?", "type": "frequency"},
            {"question": "Have you taken any medication for the cough?", "type": "medication"},
            {"question": "Do you have any other symptoms?", "type": "symptom"}
        ]
    }
    
    # Find best matching symptom
    symptom_lower = symptom.lower()
    for key in default_questions:
        if key in symptom_lower:
            return default_questions[key]
    
    # Generic questions if no match
    return [
        {"question": "How severe are your symptoms?", "type": "severity"},
        {"question": "How long have you had these symptoms?", "type": "duration"},
        {"question": "Have you taken any medication?", "type": "medication"},
        {"question": "Have you had similar symptoms before?", "type": "history"}
    ]

def get_symptom_questions(disease):
    """Generate structured symptom questions dynamically based on disease-related symptoms."""
    try:
        # First check if we have default questions available (faster than API calls)
        default_questions = get_default_symptom_questions(disease)
        if default_questions:
            return default_questions
            
        # ...existing code...
    except Exception as e:
        st.error(f"Error generating questions: {str(e)}")
        return get_default_symptom_questions(disease)  # Fallback to defaults

# ...existing code...

# Handle user input
user_input = st.chat_input("Ask your medical question...")
if user_input:
    # Display user's message first
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    try:
        # Extract medical content from potentially mixed queries
        medical_content = extract_medical_content(user_input)
        
        # First check if query is purely non-medical
        if is_non_medical_query(user_input) and not is_medical_query(medical_content):
            non_medical_response = get_non_medical_response()
            st.chat_message("assistant").write(non_medical_response)
            st.session_state.messages.append({"role": "assistant", "content": non_medical_response})
        elif is_medical_query(medical_content) and not st.session_state.collecting_symptoms:
            # Initialize symptom collection
            st.session_state.collecting_symptoms = True
            st.session_state.current_disease = medical_content
            st.session_state.symptom_questions = get_symptom_questions(medical_content)
            st.session_state.symptoms = []
            st.session_state.show_dropdowns = True
            st.session_state.submitted = False
            st.session_state.responses = {}
            st.session_state.current_question = 0  # Reset to first question

            # First acknowledge the user's query and display first question immediately
            initial_response = f"Thank you for consulting about your health concern. I'll need to ask you a few questions to better understand your condition."
            st.chat_message("assistant").write(initial_response)
            st.session_state.messages.append({"role": "assistant", "content": initial_response})
            
            # Force a rerun to display the symptom collection UI immediately
            st.rerun()
    except Exception as e:
        print(f"An error occurred: {e}")
# ...existing code...