import streamlit as st
import os
import tempfile
from PIL import Image
from datetime import datetime
from storage import store_user_directory, upload_file_tolake, list_user_files, download_file

# Configure page
st.set_page_config(
    page_title="MediLocker - One Med Hub",
    page_icon="📁",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        background-color: #f0f4f8;
    }
    .file-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid #e0e0e0;
        transition: transform 0.2s;
    }
    .file-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    .category-header {
        background-color: #1a4f72;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 20px 0 10px 0;
    }
    .stButton button {
        background-color: #1a4f72;
        color: white;
    }
    .stButton button:hover {
        background-color: #153d59;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .upload-area {
        border: 2px dashed #1a4f72;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'user_authenticated' not in st.session_state:
    st.session_state.user_authenticated = False
    st.session_state.username = None
    st.session_state.user_id = None
    st.session_state.upload_success = False
    st.session_state.upload_message = ""
    st.session_state.error_message = ""
    st.session_state.selected_category = "prescriptions"
    st.session_state.file_list = []
    st.session_state.refresh_files = True

# Mock authentication function (replace with actual authentication check)
def check_authentication():
    # In a real app, this would check the Flask session or a token
    # For this demo, we'll use a mock session
    if 'username' in st.session_state and st.session_state.username:
        return True
    return False

# Set authentication state from Flask session (would be done automatically in a real setup)
def mock_login():
    st.session_state.user_authenticated = True
    st.session_state.username = "demo_user"
    st.session_state.user_id = "12345"
    # Create user directory in Data Lake if not exists
    success, _ = store_user_directory(st.session_state.username, st.session_state.user_id)
    if not success:
        st.error("Failed to initialize storage. Please try again.")

# Main App
def main():
    st.title("📁 MediLocker")
    st.write("Securely store and access your medical documents in one place.")
    
    # Check authentication
    if not check_authentication():
        with st.container():
            st.warning("You need to log in to access MediLocker.")
            if st.button("Mock Login (Demo Only)"):
                mock_login()
                st.experimental_rerun()
        return

    # Display success or error messages if any
    if st.session_state.upload_success:
        st.markdown(f"<div class='success-message'>{st.session_state.upload_message}</div>", 
                    unsafe_allow_html=True)
        # Reset after showing
        st.session_state.upload_success = False
        st.session_state.upload_message = ""
    
    if st.session_state.error_message:
        st.markdown(f"<div class='error-message'>{st.session_state.error_message}</div>", 
                     unsafe_allow_html=True)
        # Reset after showing
        st.session_state.error_message = ""

    # Create two columns for the UI
    col1, col2 = st.columns([1, 2])
    
    # Left column: Upload section
    with col1:
        st.header("Upload Documents")
        
        # Category selection
        category = st.selectbox(
            "Select Document Category",
            ["Prescriptions", "Lab Reports", "X-Rays", "Medical History", "Diagnosis"],
            index=["prescriptions", "lab_reports", "x_rays", "medical_history", "diagnosis"]
            .index(st.session_state.selected_category)
        )
        st.session_state.selected_category = category.lower().replace(" ", "_")
        
        # File upload
        st.markdown("<div class='upload-area'>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Drop your file here or click to upload", 
                                        type=["pdf", "jpg", "jpeg", "png", "doc", "docx", "txt"])
        st.markdown("</div>", unsafe_allow_html=True)
        
        if uploaded_file:
            # Display file preview
            st.write("File Preview:")
            if uploaded_file.type.startswith('image'):
                image = Image.open(uploaded_file)
                st.image(image, width=250)
            else:
                st.write(f"File: {uploaded_file.name} ({uploaded_file.type})")
            
            # Create a temporary file to store the uploaded content
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # Upload button
            if st.button("Upload to MediLocker"):
                with st.spinner("Uploading file..."):
                    success, message = upload_file_tolake(
                        st.session_state.username,
                        st.session_state.user_id,
                        tmp_path,
                        st.session_state.selected_category,
                        uploaded_file.name
                    )
                
                # Clean up the temporary file
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                
                if success:
                    st.session_state.upload_success = True
                    st.session_state.upload_message = f"File '{uploaded_file.name}' successfully uploaded!"
                    st.session_state.refresh_files = True
                else:
                    st.session_state.error_message = f"Error uploading file: {message}"
                
                st.experimental_rerun()
    
    # Right column: File listing
    with col2:
        st.header("Your Documents")
        
        # Category tabs
        categories = ["Prescriptions", "Lab Reports", "X-Rays", "Medical History", "Diagnosis"]
        tabs = st.tabs(categories)
        
        # Fetch files for each category
        for i, tab in enumerate(tabs):
            with tab:
                category_name = categories[i].lower().replace(" ", "_")
                if st.session_state.refresh_files:
                    success, files = list_user_files(
                        st.session_state.username,
                        st.session_state.user_id,
                        category_name
                    )
                    if i == 0:  # Only need to do this once
                        st.session_state.refresh_files = False
                else:
                    success, files = list_user_files(
                        st.session_state.username,
                        st.session_state.user_id,
                        category_name
                    )
                
                if success:
                    if files:
                        for file in files:
                            with st.container():
                                st.markdown(f"<div class='file-card'>", unsafe_allow_html=True)
                                col_a, col_b = st.columns([3, 1])
                                
                                with col_a:
                                    # Extract and format file info
                                    file_name = os.path.basename(file)
                                    file_ext = os.path.splitext(file_name)[1]
                                    
                                    # Display file name and icon based on type
                                    if file_ext.lower() in ['.jpg', '.jpeg', '.png']:
                                        st.markdown(f"🖼️ **{file_name}**")
                                    elif file_ext.lower() == '.pdf':
                                        st.markdown(f"📄 **{file_name}**")
                                    elif file_ext.lower() in ['.doc', '.docx']:
                                        st.markdown(f"📝 **{file_name}**")
                                    else:
                                        st.markdown(f"📑 **{file_name}**")
                                    
                                    # Try to extract upload date from filename
                                    try:
                                        # Format: filename_YYYYMMDD_HHMMSS.ext
                                        date_str = file_name.split('_')[-2] + file_name.split('_')[-1].split('.')[0]
                                        date = datetime.strptime(date_str, "%Y%m%d%H%M%S")
                                        st.caption(f"Uploaded: {date.strftime('%B %d, %Y at %I:%M %p')}")
                                    except:
                                        pass
                                
                                with col_b:
                                    # Download button
                                    if st.button("Download", key=f"download_{file}"):
                                        try:
                                            # Create temp dir for download
                                            temp_dir = tempfile.mkdtemp()
                                            local_path = os.path.join(temp_dir, file_name)
                                            
                                            # Download the file
                                            success, message = download_file(
                                                st.session_state.username,
                                                st.session_state.user_id,
                                                f"{category_name}/{file_name}",
                                                local_path
                                            )
                                            
                                            if success:
                                                # Read file and create a download link
                                                with open(local_path, "rb") as f:
                                                    file_bytes = f.read()
                                                    st.download_button(
                                                        label="Save File",
                                                        data=file_bytes,
                                                        file_name=file_name,
                                                        mime=f"application/{file_ext[1:]}",
                                                        key=f"save_{file}"
                                                    )
                                            else:
                                                st.error(f"Error downloading file: {message}")
                                        except Exception as e:
                                            st.error(f"Error: {str(e)}")
                                
                                st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.info(f"No files found in {categories[i]}. Upload your first document!")
                else:
                    st.error(f"Error listing files: {files}")

# Run the app
if __name__ == "__main__":
    main()
