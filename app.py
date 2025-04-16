import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import numpy as np
from recommendation import cnv, dme, drusen, normal
import tempfile
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
import os

# Tensorflow Model Prediction with LIME Explanation
def model_prediction(test_image_path):
    # Load model with custom objects if needed
    try:
        model = tf.keras.models.load_model("Trained_Model.keras")
    except:
        # If there's an F1Score custom metric issue
        model = tf.keras.models.load_model("Trained_Model.keras", compile=False)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Load and preprocess image
    img = tf.keras.utils.load_img(test_image_path, target_size=(224, 224))
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x_preprocessed = preprocess_input(x.copy())
    
    # Make prediction
    predictions = model.predict(x_preprocessed)
    pred_index = np.argmax(predictions)
    
    # Generate LIME explanation
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        x[0].astype('double'), 
        lambda x: model.predict(preprocess_input(x)), 
        top_labels=5, 
        hide_color=0, 
        num_samples=1000
    )
    
    return pred_index, explanation, x[0]

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Identification"])

# Main Page
if app_mode == "Home":
    st.markdown("""
    ## **OCT Retinal Analysis Platform**
    [... rest of your home page content ...]
    """)

# About Project
elif app_mode == "About":
    st.header("About")
    st.markdown("""
    [... rest of your about page content ...]
    """)

# Prediction Page with LIME Explanation
elif app_mode == "Disease Identification":
    st.header("Welcome to the Retinal OCT Analysis Platform")
    test_image = st.file_uploader("Upload your Image:", type=['jpg', 'jpeg', 'png'])
    
    if test_image is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(test_image.name)[1]) as tmp_file:
            tmp_file.write(test_image.read())
            temp_file_path = tmp_file.name
            
        if st.button("Predict"):
            with st.spinner("Analyzing OCT Image..."):
                try:
                    result_index, explanation, original_image = model_prediction(temp_file_path)
                    class_name = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
                    st.success(f"Model Prediction: {class_name[result_index]}")
                    
                    # Display original image
                    st.subheader("Original OCT Image")
                    st.image(original_image.astype('uint8'), use_column_width=True)
                    
                    # Update the LIME explanation section in your code with this:

                    # Display LIME explanation with proper red/green colors
                    st.subheader("AI Explanation (Important Regions)")
                    st.markdown("""
                    **How to interpret**: 
                    - 🟢 Green areas strongly support the predicted diagnosis
                    - 🔴 Red areas contradict the prediction
                    - Intensity shows how strongly each region influenced the decision
                    """)
                    plt.style.use('default')  # Reset any weird style defaults
                    # Generate explanation with custom colors
                    temp, mask = explanation.get_image_and_mask(
                        explanation.top_labels[0],
                        positive_only=False,
                        num_features=10,  # Show more features for better coverage
                        hide_rest=False,
                        min_weight=0.05   # Filter out weak explanations
                    )

                    # Create custom colored explanation
                    fig, ax = plt.subplots(figsize=(8,8))
                    # Positive (green) and negative (red) contributions
                    positive_only = mask.copy()
                    positive_only[mask < 0] = 0
                    negative_only = mask.copy()
                    negative_only[mask > 0] = 0

                    # Display original image
                    ax.imshow(original_image.astype('uint8'), alpha=0.7)

                    # Overlay positive (green) and negative (red) regions
                    ax.imshow(positive_only, cmap='Greens', alpha=0.3)
                    ax.imshow(negative_only, cmap='Reds', alpha=0.3)
                    ax.axis('off')
                    st.pyplot(fig)
                    
                    # Disease-specific information
                    with st.expander("Learn More About This Condition"):
                        if result_index == 0:
                            st.write('OCT scan showing CNV with subretinal fluid.')
                            st.image(test_image)
                            st.markdown(cnv)
                        elif result_index == 1:
                            st.write('OCT scan showing DME with retinal thickening.')
                            st.image(test_image)
                            st.markdown(dme)
                        elif result_index == 2:
                            st.write('OCT scan showing drusen deposits.')
                            st.image(test_image)
                            st.markdown(drusen)
                        elif result_index == 3:
                            st.write('OCT scan showing normal retina.')
                            st.image(test_image)
                            st.markdown(normal)
                            
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(temp_file_path)
                    except:
                        pass

