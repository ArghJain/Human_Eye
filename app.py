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

#### **Welcome to the Retinal OCT Analysis Platform**

**Optical Coherence Tomography (OCT)** is a powerful imaging technique that provides high-resolution cross-sectional images of the retina, allowing for early detection and monitoring of various retinal diseases. Each year, over 30 million OCT scans are performed, aiding in the diagnosis and management of eye conditions that can lead to vision loss, such as choroidal neovascularization (CNV), diabetic macular edema (DME), and age-related macular degeneration (AMD).

##### **Why OCT Matters**
OCT is a crucial tool in ophthalmology, offering non-invasive imaging to detect retinal abnormalities. On this platform, we aim to streamline the analysis and interpretation of these scans, reducing the time burden on medical professionals and increasing diagnostic accuracy through advanced automated analysis.

---

#### **Key Features of the Platform**

- **Automated Image Analysis**: Our platform uses state-of-the-art machine learning models to classify OCT images into distinct categories: **Normal**, **CNV**, **DME**, and **Drusen**.
- **Cross-Sectional Retinal Imaging**: Examine high-quality images showcasing both normal retinas and various pathologies, helping doctors make informed clinical decisions.
- **Streamlined Workflow**: Upload, analyze, and review OCT scans in a few easy steps.

---

#### **Understanding Retinal Diseases through OCT**

1. **Choroidal Neovascularization (CNV)**
   - Neovascular membrane with subretinal fluid
   
2. **Diabetic Macular Edema (DME)**
   - Retinal thickening with intraretinal fluid
   
3. **Drusen (Early AMD)**
   - Presence of multiple drusen deposits

4. **Normal Retina**
   - Preserved foveal contour, absence of fluid or edema

---

#### **About the Dataset**

Our dataset consists of **84,495 high-resolution OCT images** (JPEG format) organized into **train, test, and validation** sets, split into four primary categories:
- **Normal**
- **CNV**
- **DME**
- **Drusen**

Each image has undergone multiple layers of expert verification to ensure accuracy in disease classification. The images were obtained from various renowned medical centers worldwide and span across a diverse patient population, ensuring comprehensive coverage of different retinal conditions.

---

#### **Get Started**

- **Upload OCT Images**: Begin by uploading your OCT scans for analysis.
- **Explore Results**: View categorized scans and detailed diagnostic insights.
- **Learn More**: Dive deeper into the different retinal diseases and how OCT helps diagnose them.


    """)

# About Project
elif app_mode == "About":
    st.header("About")
    st.markdown("""
                #### About Dataset
                Retinal optical coherence tomography (OCT) is an imaging technique used to capture high-resolution cross sections of the retinas of living patients. 
                Approximately 30 million OCT scans are performed each year, and the analysis and interpretation of these images takes up a significant amount of time.
                (A) (Far left) choroidal neovascularization (CNV) with neovascular membrane (white arrowheads) and associated subretinal fluid (arrows). 
                (Middle left) Diabetic macular edema (DME) with retinal-thickening-associated intraretinal fluid (arrows). 
                (Middle right) Multiple drusen (arrowheads) present in early AMD. 
                (Far right) Normal retina with preserved foveal contour and absence of any retinal fluid/edema.

                ---

                #### Content
                The dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (NORMAL,CNV,DME,DRUSEN). 
                There are 84,495 X-Ray images (JPEG) and 4 categories (NORMAL,CNV,DME,DRUSEN).

                Images are labeled as (disease)-(randomized patient ID)-(image number by this patient) and split into 4 directories: CNV, DME, DRUSEN, and NORMAL.

                Optical coherence tomography (OCT) images (Spectralis OCT, Heidelberg Engineering, Germany) were selected from retrospective cohorts of adult patients from the Shiley Eye Institute of the University of California San Diego, the California Retinal Research Foundation, Medical Center Ophthalmology Associates, the Shanghai First People’s Hospital, and Beijing Tongren Eye Center between July 1, 2013 and March 1, 2017.

                Before training, each image went through a tiered grading system consisting of multiple layers of trained graders of increasing exper- tise for verification and correction of image labels. Each image imported into the database started with a label matching the most recent diagnosis of the patient. The first tier of graders consisted of undergraduate and medical students who had taken and passed an OCT interpretation course review. This first tier of graders conducted initial quality control and excluded OCT images containing severe artifacts or significant image resolution reductions. The second tier of graders consisted of four ophthalmologists who independently graded each image that had passed the first tier. The presence or absence of choroidal neovascularization (active or in the form of subretinal fibrosis), macular edema, drusen, and other pathologies visible on the OCT scan were recorded. Finally, a third tier of two senior independent retinal specialists, each with over 20 years of clinical retina experience, verified the true labels for each image. The dataset selection and stratification process is displayed in a CONSORT-style diagram in Figure 2B. To account for human error in grading, a validation subset of 993 scans was graded separately by two ophthalmologist graders, with disagreement in clinical labels arbitrated by a senior retinal specialist.

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

