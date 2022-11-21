import os


# uploaded images
def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads',uploaded_file.name),'wb')as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0