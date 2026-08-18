import os
import importlib.util
import numpy as np
from PIL import Image
import cv2
import pandas as pd
from datetime import datetime
import json
from pathlib import Path
import re
from typing import List, Dict, Any, Optional, Union, Tuple

# Check if pydicom is available
pydicom_available = importlib.util.find_spec("pydicom") is not None
if pydicom_available:
    import pydicom

# Check if google.generativeai is available
genai_available = importlib.util.find_spec("google.generativeai") is not None
if genai_available:
    import google.generativeai as genai

class MedicalImageAnalyzer_x_ray:
    def __init__(self, api_key=None, gemini_model_name="gemini-1.5-flash"):
        """
        Initialize the medical image analyzer.
        
        Args:
            api_key: Google Gemini API key (required)
            gemini_model_name: Name of the Gemini model to use (default: gemini-1.5-flash)
        """
        # Gemini API initialization
        self.api_key = api_key
        self.gemini_model = None
        self.gemini_model_name = gemini_model_name
        if genai_available and api_key:
            self.initialize_gemini()
    
    def initialize_gemini(self):
        
        """Initialize the Gemini model."""
        if not genai_available:
            print("Google GenerativeAI package is not installed. Please install it with: pip install google-generativeai")
            return
            
        try:
            genai.configure(api_key=self.api_key)
            
            # Configure the model
            generation_config = {
                "temperature": 0.1,
                "top_p": 0.95,
                "top_k": 40,
            }
            
            # Initialize the Gemini model
            self.gemini_model = genai.GenerativeModel(
                model_name=self.gemini_model_name,
                generation_config=generation_config
            )
            print(f"Gemini model '{self.gemini_model_name}' initialized successfully")
        except Exception as e:
            print(f"Error initializing Gemini model: {str(e)}")
            if "404" in str(e) or "deprecated" in str(e).lower():
                print("The specified Gemini model may be deprecated or unavailable.")
                print("Try using gemini-1.5-flash, gemini-1.5-pro, or another current model.")
            elif "invalid API key" in str(e).lower():
                print("Please check that your Gemini API key is valid and not expired")
            elif "quota" in str(e).lower():
                print("You may have exceeded your Gemini API quota or rate limits")
    
    def read_medical_image(self, image_path):
        """Read medical image from various formats (DICOM, JPEG, PNG)."""
        _, ext = os.path.splitext(image_path.lower())
        
        if ext == '.dcm':
            if not pydicom_available:
                print("pydicom is not installed. Please install it to read DICOM files.")
                return None, None, None
                
            try:
                dicom_data = pydicom.dcmread(image_path)
                img_array = dicom_data.pixel_array
                
                # Normalize to 0-255 range using numpy instead of skimage
                min_val = np.min(img_array)
                max_val = np.max(img_array)
                normalized = ((img_array - min_val) * 255.0 / (max_val - min_val)).astype(np.uint8)
                
                # Create a displayable image
                display_img = Image.fromarray(normalized)
                
                return normalized, display_img, dicom_data
            
            except Exception as e:
                print(f"Error reading DICOM file: {str(e)}")
                return None, None, None
        else:
            try:
                img = Image.open(image_path)
                img_array = np.array(img)
                display_img = img.copy()
                return img_array, display_img, None
            
            except Exception as e:
                print(f"Error reading image file: {str(e)}")
                return None, None, None
    
    def analyze_medical_image(self, image_path, image_type="X-ray"):
        """
        Analyze a medical image and return diagnostic information.
        
        Args:
            image_path: Path to the medical image file
            image_type: Type of medical image (X-ray, MRI, CT scan, etc.)
            
        Returns:
            Dictionary containing the analysis results
        """
        if genai_available and self.gemini_model:
            return self.analyze_with_gemini(image_path, image_type)
        else:
            return {"error": "Gemini Vision analysis is not available. Please check your API key or install the required package with: pip install google-generativeai"}
    
    def analyze_with_gemini(self, image_path, image_type="X-ray"):
        """Analyze image using Gemini Vision."""
        try:
            # Load the image
            image = Image.open(image_path)
            
            # Enhanced prompt with more specific instructions and medical context
            prompt = f"""
            You are a skilled radiologist with expertise in analyzing {image_type} images. Please analyze this medical {image_type} image in detail.

            First, determine if this is a normal (healthy) image or if it shows abnormalities.

            Then, provide the following information in a structured format:
            1. Detailed description of what you observe in the image including anatomical structures
            2. Are there any abnormalities visible? If yes, describe them in detail
            3. Potential diagnoses based on the image findings, or confirm normal findings
            4. Confidence level in your observation (provide a percentage)
            5. Recommendations for further tests or investigations if needed

            IMPORTANT INSTRUCTIONS:
            - Be thorough and methodical in your analysis
            - Only suggest diagnoses that are consistent with visible evidence
            - Distinguish between normal variations and pathological findings
            - If the image appears normal, clearly state that no abnormalities are visible
            - Do not overdiagnose - if uncertain, express your uncertainty
            
            At the end of your response, provide a clear and specific conclusion in this exact format:
            "PREDICTION: [Most Likely Finding/Diagnosis] (Confidence: XX%)"
            - If no abnormalities, state "PREDICTION: Normal (Confidence: XX%)"
            - If abnormal, be specific about the condition, e.g., "PREDICTION: Pneumonia (Confidence: XX%)"
            """
            
            # Generate content with the image
            response = self.gemini_model.generate_content([prompt, image])
            
            # Process and structure the response
            analysis_text = response.text
            
            # Extract prediction and confidence with enhanced validation
            prediction = None
            confidence = None
            for line in analysis_text.splitlines():
                if line.startswith("PREDICTION:"):
                    try:
                        pred_line = line.replace("PREDICTION:", "").strip()
                        # Handle different formatting possibilities
                        if "Confidence:" in pred_line:
                            disease, conf = pred_line.split("(Confidence:")
                            prediction = disease.strip()
                            confidence = float(conf.replace("%", "").replace(")", "").strip()) / 100.0
                        elif "confidence:" in pred_line.lower():
                            disease, conf = pred_line.split("(confidence:")
                            prediction = disease.strip()
                            confidence = float(conf.replace("%", "").replace(")", "").strip()) / 100.0
                        else:
                            # Try to find percentage in the line
                            prediction = pred_line
                            confidence_matches = re.findall(r'(\d+)%', pred_line)
                            if confidence_matches:
                                confidence = float(confidence_matches[0]) / 100.0
                            else:
                                confidence = 0.7  # Default confidence if parsing fails
                    except Exception as e:
                        print(f"Error parsing prediction: {str(e)}")
                        # If parsing fails, still try to extract the disease name
                        prediction = pred_line if 'pred_line' in locals() else "Unknown"
                        confidence = 0.5  # Default confidence
                    break
            
            # Validate the prediction
            if prediction and "normal" in prediction.lower() and "not normal" not in prediction.lower():
                # Make sure if it says "normal" it doesn't also mention a disease
                common_diseases = ["pneumonia", "cancer", "tumor", "fracture", "cardiomegaly", 
                                  "edema", "effusion", "pneumothorax", "atelectasis", "emphysema"]
                
                contains_disease = False
                for disease in common_diseases:
                    if disease in prediction.lower():
                        contains_disease = True
                        break
                
                if contains_disease:
                    # If prediction says "normal" but also mentions a disease, clarify
                    prediction = "Abnormal findings - see detailed analysis"
            
            # Create a structured output
            result = {
                "raw_analysis": analysis_text,
                "image_type": image_type,
                "timestamp": datetime.now().isoformat(),
                "filename": os.path.basename(image_path),
                "analysis_method": f"Gemini {self.gemini_model_name}",
                "disclaimer": "This analysis is generated by AI and should not replace professional medical diagnosis.",
                "prediction": prediction if prediction else "Analysis completed - see detailed report",
                "confidence": confidence if confidence else 0.5
            }
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            if "404" in error_msg or "deprecated" in error_msg.lower():
                return {"error": f"The Gemini model '{self.gemini_model_name}' is deprecated or unavailable. Try using gemini-1.5-flash or gemini-1.5-pro."}
            else:
                return {"error": error_msg}
    
    def batch_analyze(self, image_directory, image_type="X-ray"):
        """Analyze multiple medical images in a directory."""
        results = []
        image_paths = [f for f in Path(image_directory).glob("*") 
                      if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tiff', '.dcm']]
        
        for image_path in image_paths:
            result = self.analyze_medical_image(str(image_path), image_type)
            results.append(result)
            
        return results

# Example usage
if __name__ == "__main__":
    # For Gemini Vision (example, replace with actual API key)
    API_KEY = "YOUR_GEMINI_API_KEY"
    
    # Initialize with API key
    analyzer = MedicalImageAnalyzer_x_ray(api_key=API_KEY)
    
    # Example: Analyze a single image with Gemini
    # image_path = "path/to/xray_image.jpg"
    # result_gemini = analyzer.analyze_medical_image(image_path, image_type="Chest X-ray")
    # analyzer.visualize_with_analysis(image_path, result_gemini)
    
    # Example: Batch analyze multiple images
    # results = analyzer.batch_analyze("path/to/image_directory", image_type="CT scan")