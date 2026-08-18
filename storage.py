from config import CONNECTION_STRING, FILE_SYSTEM_NAME
from azure.storage.filedatalake import DataLakeServiceClient
import os
from datetime import datetime
service_client = DataLakeServiceClient.from_connection_string(CONNECTION_STRING)
file_system_client = service_client.get_file_system_client(FILE_SYSTEM_NAME)

user_container = service_client.get_file_system_client("user-data")

def store_user_directory(username, user_id):
    """Create main user directory and standard subdirectories"""
    folder_name = f"{username}_{user_id}"
    # Standard subdirectories for user data
    subdirs = ["prescriptions", "lab_reports", "x_rays", "medical_history", "diagnosis"]
    
    try:
        # Create main directory if it doesn't exist
        directory_client = user_container.get_directory_client(folder_name)
        directory_client.create_directory()
        
        # Create subdirectories
        for subdir in subdirs:
            subdir_client = user_container.get_directory_client(f"{folder_name}/{subdir}")
            subdir_client.create_directory()
        
        return True, folder_name
    except Exception as e:
        print(f"Error creating directory: {str(e)}")
        return False, str(e)

def upload_file_tolake(username, user_id, file_path, subdir, file_name=None):
    """
    Upload a file to a user's subdirectory in Azure Data Lake
    
    Parameters:
    - username: User's username
    - user_id: User's unique ID
    - file_path: Path to the local file to upload
    - subdir: Subdirectory name (e.g., 'prescriptions', 'x_rays')
    - file_name: Optional custom file name, otherwise uses original file name
    
    Returns:
    - (success, message) tuple
    """
    try:
        # Normalize the subdir name and validate
        subdir = subdir.lower().strip()
        valid_subdirs = ["prescriptions", "lab_reports", "x_rays", "medical_history", "diagnosis"]
        
        if subdir not in valid_subdirs:
            return False, f"Invalid subdirectory: {subdir}. Valid options are: {', '.join(valid_subdirs)}"
        
        # Get file name if not provided
        if not file_name:
            file_name = os.path.basename(file_path)
        
        # Add timestamp to make filename unique
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = os.path.splitext(file_name)[1]
        base_name = os.path.splitext(file_name)[0]
        unique_file_name = f"{base_name}_{timestamp}{file_extension}"
        
        # Construct the full path in the data lake
        folder_name = f"{user_container}_{user_id}"
        file_path_in_lake = f"{folder_name}/{subdir}/{unique_file_name}"
        
        # Create file client
        file_client = user_container.get_file_client(file_path_in_lake)
        
        # Upload the file
        with open(file_path, "rb") as data:
            file_client.upload_data(data, overwrite=True)
        
        return True, file_path_in_lake
        
    except FileNotFoundError:
        return False, f"File not found: {file_path}"
    except Exception as e:
        return False, f"Error uploading file: {str(e)}"

def list_user_files(username, user_id, subdir=None):
    """
    List files in a user's directory or subdirectory
    
    Parameters:
    - username: User's username
    - user_id: User's unique ID
    - subdir: Optional subdirectory name, if None lists all files
    
    Returns:
    - (success, file_list or error_message)
    """
    try:
        folder_name = f"{username}_{user_id}"
        
        # If subdirectory specified, list only those files
        if subdir:
            path = f"{folder_name}/{subdir}"
        else:
            path = folder_name
            
        directory_client = user_container.get_directory_client(path)
        
        # Get list of files
        file_list = []
        paths = directory_client.get_paths(recursive=True)
        
        for path in paths:
            if not path.is_directory:
                # Extract just the filename from the full path
                full_path = path.name
                if subdir:
                    # If we're listing a specific subdir, the path will be like "username_id/subdir/file.ext"
                    # We just want "file.ext"
                    file_list.append(os.path.basename(full_path))
                else:
                    # If we're listing all files, the path will be like "username_id/subdir/file.ext"
                    # We want "subdir/file.ext"
                    file_list.append(full_path.replace(f"{folder_name}/", ""))
        
        return True, file_list
    
    except Exception as e:
        return False, f"Error listing files: {str(e)}"

def download_file(username, user_id, file_path, local_path):
    """
    Download a file from a user's directory
    
    Parameters:
    - username: User's username
    - user_id: User's unique ID
    - file_path: Path to the file in the data lake (relative to user directory)
    - local_path: Local path where to save the file
    
    Returns:
    - (success, message) tuple
    """
    try:
        folder_name = f"{username}_{user_id}"
        full_path = f"{folder_name}/{file_path}"
        
        # Create file client
        file_client = user_container.get_file_client(full_path)
        
        # Download the file
        with open(local_path, "wb") as local_file:
            download = file_client.download_file()
            local_file.write(download.readall())
        
        return True, local_path
    
    except Exception as e:
        return False, f"Error downloading file: {str(e)}"
    

print(store_user_directory("jms", "jms_onemeduser1000"))
print(upload_file_tolake("jms", "jms_onemeduser1000", r"C:\Users\jeffr\OneDrive\Desktop\onemed\templates\pres.html", "prescriptions"))