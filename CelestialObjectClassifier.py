import threading
import streamlit as st
import numpy as np
import pandas as pd
import time
from streamlit.components.v1 import html
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

class DataTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['u_corrected'] = X['u'] - X['extinction_u']
        X['g_corrected'] = X['g'] - X['extinction_g']
        X['r_corrected'] = X['r'] - X['extinction_r']
        X['i_corrected'] = X['i'] - X['extinction_i']
        X['z_corrected'] = X['z'] - X['extinction_z']

        X['u-g'] = X['u_corrected'] - X['g_corrected']
        X['g-r'] = X['g_corrected'] - X['r_corrected']
        X['r-i'] = X['r_corrected'] - X['i_corrected']
        X['i-z'] = X['i_corrected'] - X['z_corrected']

        X = X.drop(['u', 'g', 'r', 'i', 'z', 'extinction_u', 'extinction_g', 'extinction_r', 'extinction_i', 'extinction_z'], axis=1)
        return X

class CombinePipelines(BaseEstimator, TransformerMixin):
    def __init__(self, preprocessing_pipeline, final_pipeline):
        self.preprocessing_pipeline = preprocessing_pipeline
        self.final_pipeline = final_pipeline

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_preprocessed = self.preprocessing_pipeline.transform(X)
        return self.final_pipeline.transform(X_preprocessed)

    def predict(self, X):
        X_preprocessed = self.preprocessing_pipeline.transform(X)
        return self.final_pipeline.predict(X_preprocessed)

    def predict_proba(self, X):
        X_preprocessed = self.preprocessing_pipeline.transform(X)
        return self.final_pipeline.predict_proba(X_preprocessed)

celestial_object_classifier = joblib.load('./celestial_object_classifier.pkl')

def predict_celestial_object(features):
    return celestial_object_classifier.predict(features)



st.set_page_config(
    page_title="Celestial Object Classifier",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide streamlit's default header
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Particles HTML Implementation
particles_html = """
<html>
<head>
   <script src="https://cdnjs.cloudflare.com/ajax/libs/tsparticles/1.18.11/tsparticles.min.js"></script>
   <style>
      #particles {
         position: fixed;
         width: 100%;
         height: 100%;
         top: 0;
         left: 0;
         z-index: 0;
         background: linear-gradient(180deg, #0a0a2c 0%, #162447 100%);
      }
   </style>
</head>
<body>
   <div id="particles"></div>
   <script>
      tsParticles.load("particles", {
         particles: {
            number: {
                value: 100,
                density: {
                    enable: true,
                    value_area: 800
                }
            },
            color: {
                value: "#ffffff"
            },
            shape: {
                type: "circle"
            },
            opacity: {
                value: 0.5,
                random: true,
                anim: {
                    enable: true,
                    speed: 1,
                    opacity_min: 0.1,
                    sync: false
                }
            },
            size: {
                value: 3,
                random: true,
                anim: {
                    enable: true,
                    speed: 2,
                    size_min: 0.1,
                    sync: false
                }
            },
            line_linked: {
                enable: true,
                distance: 150,
                color: "#ffffff",
                opacity: 0.2,
                width: 1
            },
            move: {
                enable: true,
                speed: 1,
                direction: "none",
                random: true,
                straight: false,
                out_mode: "out",
                bounce: false
            }
         },
         interactivity: {
            detect_on: "canvas",
            events: {
                onhover: {
                    enable: true,
                    mode: "grab"
                },
                onclick: {
                    enable: true,
                    mode: "push"
                },
                resize: true
            },
            modes: {
                grab: {
                    distance: 140,
                    line_linked: {
                        opacity: 0.5
                    }
                },
                push: {
                    particles_nb: 4
                }
            }
         },
         retina_detect: true
      });
   </script>
</body>
</html>
"""

# Inject particles
html(particles_html, height=1000)

# Main CSS
st.markdown("""
<style>
    .stApp {
        background: transparent !important;
    }
    
    iframe {
        position: fixed;
        left: 0;
        right: 0;
        top: 0;
        bottom: 0;
        width: 100vw;
        height: 100vh;
        pointer-events: none;
    }
    
    .main {
        position: relative;
        z-index: 1;
    }
    
    .title {
        font-size: 3.5em;
        font-weight: bold;
        text-align: center;
        color: white;
        margin-bottom: 1em;
        text-shadow: 0 0 10px rgba(0, 255, 157, 0.5),
                     0 0 20px rgba(0, 255, 157, 0.3),
                     0 0 30px rgba(0, 255, 157, 0.2);
        position: relative;
        z-index: 2;
    }
    
    .cosmic-card {
        background: rgba(13, 27, 72, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        padding: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        margin: 10px 0;
        transition: transform 0.3s ease;
        position: relative;
        z-index: 2;
        height: 100%;
        color: green;
    }
    
    .cosmic-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.47);
    }
    
    .stNumberInput input {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 8px !important;
        color: black !important;
        padding: 12px !important;
    }
    
    .stNumberInput input:focus {
        border-color: #00ff9d !important;
        box-shadow: 0 0 20px rgba(0, 255, 157, 0.3) !important;
    }
    
    .classify-button-container {
        display: flex;
        justify-content: center;   /* Horizontally centers the button */
        align-items: center;       /* Vertically center if needed */
        width: 100%;               /* Container takes the full width */
        margin: 30px 0;            /* Adjust top and bottom margin */
        position: relative;
        z-index: 2;
    }

    .stButton {
        display: flex !important;
        justify-content: center !important;   /* Centers the button inside the stButton container */
        width: 100% !important;               /* Ensures full-width container */
    }

    .stButton > button {
        background: linear-gradient(45deg, #00ff9d, #00f0ff) !important;
        color: black !important;
        padding: 15px 50px !important;
        border-radius: 12px !important;
        border: none !important;
        font-weight: bold !important;
        font-size: 1.2em !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        width: auto !important;
        margin: 0 auto !important;   /* Centers the button inside */
    }

    .stButton > button:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 0 25px rgba(0, 255, 157, 0.5) !important;
    }
    
    .loading-text {
        background: linear-gradient(90deg, #00ff9d, #00f0ff, #0066ff, #00ff9d);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientText 3s ease infinite;
        font-size: 1.5em;
        text-align: center;
    }
    
    @keyframes resultAppear {
        0% { transform: scale(0.9); opacity: 0; }
        100% { transform: scale(1); opacity: 1; }
    }
            
        .group-info {
        background: rgba(0, 0, 0, 0.3);
        border-radius: 15px;
        padding: 20px;
        margin: 20px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        position: relative;
        z-index: 2;
    }
    
    .member-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 10px 15px;
        margin: 5px 0;
        transition: all 0.3s ease;
    }
    
    .member-card:hover {
        background: rgba(255, 255, 255, 0.1);
        transform: translateX(10px);
    }
    
    @keyframes gradientText {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .result-container {
        animation: resultAppear 0.8s ease-out forwards;
    }
    
    .parameter-description {
        color: rgba(255, 255, 255, 0.7);
        font-size: 0.9em;
        margin-bottom: 10px;
    }
    
    .stNumberInput label {
            color: #ffcc00;  /* Change to your preferred color */
            font-weight: bold;  /* Optional: Make the text bold */
            font-size: 18px;  /* Optional: Adjust font size */
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title'>🌌 Celestial Object Classifier</h1>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<div class='cosmic-card'><h3>📊 Spectral Data</h3></div>", unsafe_allow_html=True)
    
    st.markdown("<div class='parameter-description'>Ultraviolet filter - Measures high-energy emissions</div>", unsafe_allow_html=True)
    u = st.number_input("Ultraviolet (u)", value=0.0, format="%.6f", key="u")
    
    st.markdown("<div class='parameter-description'>Green filter - Visible spectrum measurements</div>", unsafe_allow_html=True)
    g = st.number_input("Green (g)", value=0.0, format="%.6f", key="g")
    
    st.markdown("<div class='parameter-description'>Red filter - Detects cooler stars</div>", unsafe_allow_html=True)
    r = st.number_input("Red (r)", value=0.0, format="%.6f", key="r")
    
    st.markdown("<div class='parameter-description'>Near Infrared - Studies dust-obscured objects</div>", unsafe_allow_html=True)
    i = st.number_input("Near Infrared (i)", value=0.0, format="%.6f", key="i")
    
    st.markdown("<div class='parameter-description'>Infrared filter - Observes distant objects</div>", unsafe_allow_html=True)
    z = st.number_input("Infrared (z)", value=0.0, format="%.6f", key="z")

with col2:
    st.markdown("<div class='cosmic-card'><h3>🌫️ Extinction Values</h3></div>", unsafe_allow_html=True)
    
    st.markdown("<div class='parameter-description'>UV light absorption by dust</div>", unsafe_allow_html=True)
    ext_u = st.number_input("Extinction u", value=0.0, format="%.6f", key="ext_u")
    
    st.markdown("<div class='parameter-description'>Green light absorption by dust</div>", unsafe_allow_html=True)
    ext_g = st.number_input("Extinction g", value=0.0, format="%.6f", key="ext_g")
    
    st.markdown("<div class='parameter-description'>Red light absorption by dust</div>", unsafe_allow_html=True)
    ext_r = st.number_input("Extinction r", value=0.0, format="%.6f", key="ext_r")
    
    st.markdown("<div class='parameter-description'>Near-infrared light absorption</div>", unsafe_allow_html=True)
    ext_i = st.number_input("Extinction i", value=0.0, format="%.6f", key="ext_i")
    
    st.markdown("<div class='parameter-description'>Infrared light absorption</div>", unsafe_allow_html=True)
    ext_z = st.number_input("Extinction z", value=0.0, format="%.6f", key="ext_z")

with col3:
    st.markdown("<div class='cosmic-card'><h3>🌟 Additional Information</h3></div>", unsafe_allow_html=True)
    
    st.markdown("<div class='parameter-description'>Right Ascension - celestial longitude</div>", unsafe_allow_html=True)
    ra = st.number_input("Right Ascension (z)", value=0.0, format="%.6f", key="ra")
    
    st.markdown("<div class='parameter-description'>Declination - celestial latitude</div>", unsafe_allow_html=True)
    dec = st.number_input("Declination (z)", value=0.0, format="%.6f", key="dec")
    
    st.markdown("<div class='parameter-description'>Indicates object's velocity and distance</div>", unsafe_allow_html=True)
    redshift = st.number_input("Redshift", value=0.0, format="%.6f", key="redshift")
    


loading = False
result = None


def show_loading(loading_placeholder):
    """Display loading animation."""
    global loading
    
    while loading:
        for message in loading_messages:
            print("Loading")
            if not loading:
                break
            loading_placeholder.markdown(
                f"<div class='loading-text'>{message}</div>",
                unsafe_allow_html=True
            )
            time.sleep(0.3)


def process_model(features):
    """Run the model and update result."""
    global loading, result
    result = predict_celestial_object(features)
    loading = False


def validate_inputs(ra, dec, u, g, r, i, z, redshift, ext_u, ext_g, ext_r, ext_i, ext_z):
    if all(v == 0 for v in [ra, dec, u, g, r, i, z, redshift, ext_u, ext_g, ext_r, ext_i, ext_z]):
        return "Invalid input: All input values cannot be zero."
    if all(v == 0 for v in [u, g, r, i, z]):
        return "Invalid input: All filter magnitudes (u, g, r, i, z) cannot be zero at the same time."
    if all(v == 0 for v in [ext_u, ext_g, ext_r, ext_i, ext_z]):
        return "Invalid input: All extinction values (extinction_u, extinction_g, extinction_r, extinction_i, extinction_z) cannot be zero at the same time."
    if ra == 0 and dec == 0:
        return "Invalid input: Both Right Ascension (ra) and Declination (dec) cannot be zero together."
    
    if not (0 <= ra < 360):
        return "Invalid input: Right Ascension (ra) must be between 0 and 360 degrees."
    if not (-90 <= dec <= 90):
        return "Invalid input: Declination (dec) must be between -90 and 90 degrees."
    if not (-30 <= u <= 30):
        return "Invalid input: Ultraviolet filter (u) magnitude must be between -30 and 30."
    if not (-30 <= g <= 30):
        return "Invalid input: Green filter (g) magnitude must be between -30 and 30."
    if not (-30 <= r <= 30):
        return "Invalid input: Red filter (r) magnitude must be between -30 and 30."
    if not (-30 <= i <= 30):
        return "Invalid input: Near Infrared filter (i) magnitude must be between -30 and 30."
    if not (-30 <= z <= 30):
        return "Invalid input: Infrared filter (z) magnitude must be between -30 and 30."
    if not (0 <= redshift):
        return "Invalid input: Redshift must be a positive value."
    if not (0 <= ext_u <= 1):
        return "Invalid input: Extinction (u) must be between 0 and 1."
    if not (0 <= ext_g <= 1):
        return "Invalid input: Extinction (g) must be between 0 and 1."
    if not (0 <= ext_r <= 1):
        return "Invalid input: Extinction (r) must be between 0 and 1."
    if not (0 <= ext_i <= 1):
        return "Invalid input: Extinction (i) must be between 0 and 1."
    if not (0 <= ext_z <= 1):
        return "Invalid input: Extinction (z) must be between 0 and 1."

    return None

st.markdown("<div class='classify-button-container'>", unsafe_allow_html=True)
if st.button("Classify Celestial Object"):
    # Gather inputs
    column_names = ['ra', 'dec', 'u', 'g', 'r', 'i', 'z', 'redshift', 'extinction_u', 'extinction_g', 'extinction_r', 'extinction_i', 'extinction_z']
    features = [ra, dec, u, g, r, i, z, redshift, ext_u, ext_g, ext_r, ext_i, ext_z]

    # Validate inputs
    error_message = validate_inputs(ra, dec, u, g, r, i, z, redshift, ext_u, ext_g, ext_r, ext_i, ext_z)
    
    if error_message:
        st.error(error_message)
    else:
        features = pd.DataFrame(np.array(features).reshape(1, -1), columns=column_names)

        loading = True
        loading_placeholder = st.empty()

        loading_messages = [
            "🌠 Analyzing spectral signatures...",
            "🌌 Processing quantum indicators...",
            "✨ Calculating celestial metrics...",
            "🔭 Consulting astronomical databases...",
            "🛰️ Calibrating space-time coordinates...",
            "⚡ Processing quantum fluctuations...",
        ]

        threading.Thread(target=process_model, args=(features,), daemon=True).start()

        i = 0
        while loading:
            i = (i + 1) % len(loading_messages)
            loading_placeholder.markdown(
                f"<div class='loading-text'>{loading_messages[i]}</div>",
                unsafe_allow_html=True
            )
            time.sleep(0.5)

        loading_placeholder.empty()

        class_mapping = {0: 'Galaxy', 1: 'Quasar', 2: 'Star'}
        prediction = class_mapping[result[0]]
        st.markdown(f"""
            <div class='cosmic-card result-container'>
                <h2 style='text-align: center; color: #00ff9d; margin-bottom: 20px;'>
                    Classification Result
                </h2>
                <h1 style='text-align: center; color: white; font-size: 3.5em; margin-bottom: 20px;'>
                    {prediction}
                </h1>
            </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)



st.markdown("""
<div class='group-info'>
    <h3 style='text-align: center; color: #00ff9d; margin-bottom: 15px; font-size: 1.5em;'>     Group Members</h3>
    <div class='member-card'>
        <p style='text-align: center; color: white; margin: 0;'>M. Asad Tariq (21L-5266)</p>
    </div>
    <div class='member-card'>
        <p style='text-align: center; color: white; margin: 0;'>Mudesser Ahmad (21L-5387)</p>
    </div>
    <div class='member-card'>
        <p style='text-align: center; color: white; margin: 0;'>Abdul Hadi (21L-7747)</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; padding: 20px; margin-top: 30px; 
           color: rgba(255, 255, 255, 0.7); position: relative; z-index: 2;'>
    <p>Algorithm: Random Forest | Data Source: SDSS Sky Server</p>
</div>
""", unsafe_allow_html=True)