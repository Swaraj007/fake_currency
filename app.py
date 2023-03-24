import streamlit as st
import hashlib
import sqlite3
import numpy as np
from keras.models import load_model
from PIL import Image


import base64


st.set_page_config(
    page_title="Fake Currency Detection",
    page_icon=":smiley:",
    layout="wide",
    initial_sidebar_state="expanded",
)

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('bgg.jpg')  


st.sidebar.title("Fake Currency Detection")
st.sidebar.image("s1.jpg")

# Create a connection to the database
conn = sqlite3.connect('users.db')
c = conn.cursor()

# Create a table to store user information
c.execute('''CREATE TABLE IF NOT EXISTS users (username text, password text)''')

def fake_currency(imgg):
    IMAGE_SIZE = 64
    model = load_model('currency_model.h5')
    img = Image.open(imgg)

    img = img.resize((IMAGE_SIZE,IMAGE_SIZE))
    img = np.array(img)

    img = img.reshape(1,IMAGE_SIZE,IMAGE_SIZE,3)

    img = img.astype('float32')
    img = img / 255.0
    prediction = model.predict(img)
    Fake=np.argmax(prediction)

    return Fake

def hash_password(password):
    """Hash a password using SHA-256"""
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def signup():
    st.write("Create a new account")
    username = st.text_input("Enter a username")
    password = st.text_input("Enter a password", type="password")
    confirm_password = st.text_input("Confirm your password", type="password")
    
    
    signup_button = st.button("SignUp")
    st.info("Login if already have account")
    
    if signup_button:
        if password != confirm_password:
            st.error("Passwords do not match")
            return
        hashed_password = hash_password(password)
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
        conn.commit()
        st.success("You have successfully created an account. Go to login page")

def login():
    st.write("Login to your account")
    username = st.text_input("Enter your username")
    password = st.text_input("Enter your password", type="password")
    login_button = st.button("Login")
    
    if login_button:
        hashed_password = hash_password(password)
        c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, hashed_password))
        user = c.fetchone()
        if user:
            st.success("You have successfully logged in")
            session_id = user[0] # Use the username as the session ID
            st.session_state['session_id'] = session_id
        
            st.experimental_rerun()
        else:
            st.error("Incorrect username or password")
   
def get_image():
    img = st.file_uploader("Upload Image", type=["png","jpg","svg","jpeg"])
    if img:
        st.image(img, width=500)  
        result = fake_currency(img)
        
        
        if result == 0:
            st.markdown('<p style="color:lime; font-size: 30px;font-weight:bold">No Fake Currency detected</p>', unsafe_allow_html=True)
        elif result == 1:
            st.markdown('<p style="color:red; font-size: 30px;font-weight:bold">Fake Currency detected</p>', unsafe_allow_html=True)

def logout():
    st.session_state.pop('session_id', None)
    st.info("You have been logged out") 
    st.experimental_rerun() 
    
def about():
    
    para = '''<p>
    Fake currency detection is an important task in the field of image processing and computer vision. With the increasing availability of high-quality printing technology and sophisticated counterfeiting techniques, the detection of fake currency has become increasingly challenging. In this article, we will explore how we can use <span style="color: cyan; font-weight: bold;">Convolutional Neural Networks (CNNs)</span> and <span style="color: cyan; font-weight: bold;">Streamlit</span> to develop a fake currency detection system that can detect fake currency with high accuracy.
The first step in developing a fake currency detection system is to collect a dataset of currency images. We will need to collect images of both real and fake currency notes to train our CNN model. We can use web scraping techniques to collect images of real currency notes from various sources, such as the websites of central banks, financial institutions, and government agencies. For fake currency notes, we can simulate counterfeiting techniques to generate images that resemble real currency notes. We can also use publicly available datasets of fake currency notes.
Once we have collected our dataset of currency images, we can begin to train our CNN model. We can use popular deep learning libraries such as <span style="color: yellow; font-weight: bold;">TensorFlow</span> or <span style="color: yellow; font-weight: bold;">PyTorch</span> to develop and train our model. We will need to pre-process our images by resizing them to a uniform size and normalizing the pixel values to improve the performance of our model.
We can use a variety of CNN architectures such as <span style="color: yellow; font-weight: bold;">VGG, ResNet,</span> or <span style="color: yellow; font-weight: bold;">Inception</span> to develop our model. These architectures have been proven to be highly effective for image classification tasks. We can fine-tune these pre-trained models on our dataset of currency images using transfer learning. Transfer learning involves using a pre-trained model as a starting point and fine-tuning it on a new task by updating the weights of the last few layers of the model.
After training our CNN model, we can evaluate its performance on a separate validation dataset to determine its accuracy. We can also use techniques such as data augmentation to improve the performance of our model by generating additional training data. Data augmentation involves applying transformations such as rotations, flips, and shears to our training images to create new variations of the original images.
Once we have developed a CNN model that can accurately detect fake currency notes, we can integrate it into a Streamlit web application. Streamlit is a powerful tool for building data science applications and data visualizations with Python. We can use Streamlit to develop a user-friendly web application that allows users to upload images of currency notes and get real-time predictions of whether they are real or fake.
To develop our Streamlit app, we will need to create a user interface that allows users to upload their images and view the model predictions. We can use Streamlit widgets such as file_uploader and image to allow users to upload their images and display the predictions. We can also add additional features such as a histogram of pixel values and an explanation of how the model works to enhance the user experience.
In summary, fake currency detection is an important task that can be tackled using CNNs and Streamlit. By collecting a dataset of currency images, training a CNN model on this dataset, and integrating it into a Streamlit web application, we can develop a powerful tool for detecting fake currency with high accuracy. With the increasing availability of high-quality printing technology and sophisticated counterfeiting techniques, the need for such tools has never been greater. <a href="https://ieeexplore.ieee.org/document/8994968)" style="color: lime; font-weight: bold; text-decoration:none;">Know More</a></p>
    '''
    
    st.markdown(para, unsafe_allow_html=True)
    
    
    
menu = ["Upload Image","About","Signup","Login", "Logout"]
if 'session_id' not in st.session_state:
    choice = st.sidebar.selectbox("Select an option", menu[2:-1])
else:
    choice = st.sidebar.selectbox("Select an option", menu)
    
####    
st.sidebar.text("Created By :-")
st.sidebar.write("- Swaraj Sawarkar")
st.sidebar.write("- Sumit Mule")
st.sidebar.write("- Prutha Wankhede")
st.sidebar.write("- Mayuri Savikar")
st.sidebar.write("- Rushikesh Sharma")

if choice == "Login":
    login()
elif choice == "Signup":
    signup()
elif choice == "Logout":
    logout()
elif choice == "Upload Image":
    get_image()
elif choice == "About":
    about()
else:
    st.write("Welcome back, " + st.session_state['session_id'])
