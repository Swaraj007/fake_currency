import streamlit as st
import hashlib
import sqlite3
import numpy as np
from keras.models import load_model
from PIL import Image

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

    if Fake == 0:
        cd="No Fake Currency detected"

    elif Fake == 1:
        cd="Fake Currency detected"
    return cd

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
        
                
        else:
            st.error("Incorrect username or password")
   
def get_image():
    img = st.file_uploader("Upload Image", type=["png","jpg","svg","jpeg"])
    if img:
        st.image(img, width=500)  
        result = fake_currency(img)
        
        st.header(result)

def logout():
    st.session_state.pop('session_id', None)
    st.info("You have been logged out")  
    
menu = ["Upload Image","Signup","Login", "Logout"]
if 'session_id' not in st.session_state:
    choice = st.sidebar.selectbox("Select an option", menu[1:-1])
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
else:
    st.write("Welcome back, " + st.session_state['session_id'])
