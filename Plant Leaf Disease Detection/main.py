import streamlit as st
import tensorflow as tf
import numpy as np

#Tensorflow Model Predection
def model_prediction (test_image):
    model = tf.keras.models.load_model('trained_model.keras') #Loading the trained model

    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128)) #Preprocessing done on the image to convert it to (128,128) size
    input_arr = tf.keras.preprocessing.image.img_to_array(image) #Image information is converted into array
    input_arr = np.array([input_arr]) #Done to covert a single image to a batch as we trained the model on 32 batches

    prediction = model.predict(input_arr) #give the probablity of the image belonging to each class
    result_index = np.argmax(prediction) #extract the index of the class with the highest probablity

    return result_index

#sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "Disease Detection", "About"])

#home page
if (app_mode == "Home"):
    st.header("PLANT LEAF DISEASE DETECTION SYSTEM")
    #image_path ="home_page.jpeg"
    #st.image(image_path, use_column_width = True)
    st.markdown ("""
    Welcome to the Plant Leaf Disease Detection System! 🌿🔍
    
    Our mission is to help farmers in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload an Image:** Go to the **Disease Detection** page and upload an image of a plant which is suspected to be diseased.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify any potential disease.
    3. **Results:** View the results and recommendations for any further actions.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our **Plant Leaf Disease Detection System**!

    ### About Us
    Learn more about the project and our goals on the **About** page.
    """)

elif (app_mode == "About"):
    st.header("About")
    st.markdown ("""
    ### About Dataset
    This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
    This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
    A new directory containing 33 test images is created later for prediction purpose.
                
    #### Content
    1. train (70295 images)
    2. test (33 images)
    3. validation (17572 images)
                
                 
    ### Future Scope
    To further enhance the capabilities and impact of the **Plant Disease Detection System**, the following areas are identified for future development:

    ##### 1. Model Optimization:
    -	**Deeper Architectures:** Explore advanced CNN architectures like ResNet, DenseNet, or EfficientNet to improve accuracy and handle more complex datasets.
    -	**Transfer Learning:** Leverage pre-trained models to fine-tune the system for plant disease detection, reducing training time and increasing efficiency.
    -	**Improved Data Augmentation:** Employ advanced augmentation techniques such as colour jittering, random cropping, and flipping to increase dataset diversity and enhance model robustness.
                 
    ##### 2. Transition to a Fully Functional Mobile Application:
    -	Convert the web-based application into a fully functional native mobile app for both Android and iOS platforms, allowing users to have immediate access to the disease detection features directly on their smartphones.
    -	A native mobile app will provide improved performance, offline accessibility, and better integration with hardware such as cameras and push notifications for real-time alerts, making the app more effective and user-friendly.

    ##### 3. Dataset Expansion:
    -	**Diverse Datasets:** Incorporate datasets representing diverse plant species, environmental conditions, and imaging scenarios to improve the model’s generalizability.
    -	**Synthetic Data:** Utilize techniques like Generative Adversarial Networks (GANs) to generate synthetic images for rare diseases, balancing the dataset.
    -	**Expanded Data Sources:** Partner with agricultural research organizations to incorporate diverse datasets representing various plant species, regions, and environmental conditions.
                 
    ##### 4.  AI-Powered Disease Analysis:
    -	**Objective:** Implement AI-based algorithms to analyze patterns in crop images and environmental data for predicting disease outbreaks.
    -	**Benefit:** AI-driven insights could proactively alert farmers to potential risks, enabling pre-emptive measures to prevent large-scale infestations.

    ##### 5. Enhanced Resilience:
    -	**Offline Functionality:** Enable the system to operate in areas with limited or no internet connectivity by storing data locally and synchronizing it once connectivity is restored.
    -	**Error Handling:** Incorporate mechanisms to handle low-quality or unclear images effectively, ensuring consistent performance in real-world scenarios.

    ##### 6. Multi-Language Support:
    -	Add support for multiple languages to make the application accessible to a global audience, particularly in regions where English is not the primary language.

    ##### 7.  Continuous User Feedback Integration:
    -	**Objective:** Regularly collect user feedback to refine the application’s features, user interface, and overall functionality.
    -	**Benefit:** By addressing user pain points and incorporating suggestions, the application will evolve to meet the dynamic needs of its users, ensuring sustained engagement and satisfaction.

    ##### 8. Scalability and Global Expansion:
    -	**Objective:** Scale the application to support users across different countries, ensuring compatibility with local agricultural practices, regulations, and disease profiles.
    -	**Benefit:** A globally scalable solution will provide farmers with a standardized tool for disease detection, adaptable to various climates and farming systems, driving global agricultural efficiency.

                 
    ### Results & Discussion
    ##### Model Evaluation:
    The model for plant disease detection was created using a Convolutional Neural Network (CNN) and trained on a labelled dataset of plant leaf images. The training history indicates steady improvements in both accuracy and loss over the course of 10 epochs.
                 
    **Training Results:**
    -	**Training Loss:** The loss decreased significantly from 0.8131 in the first epoch to 0.0532 in the final epoch.
    -	**Training Accuracy:** The accuracy improved steadily, beginning at 74.81% in the first epoch and reaching 98.32% by the tenth epoch.
                 
    **Validation Results:**
    -	**Validation Loss:** The validation loss reduced from 0.4580 in the first epoch to 0.0881 by the tenth epoch.
    -	**Validation Accuracy:** The validation accuracy started at 85.40% in the first epoch and increased to 97.46% by the final epoch.
                 
    These results demonstrate the model’s effectiveness in distinguishing between healthy and diseased leaves with high accuracy and minimal error.

                 
    ##### Training Vs. Validation Performance:
    The trends in the training and validation metrics indicate effective learning without significant overfitting. The close alignment of training and validation accuracy and loss suggests that the model generalizes well to unseen data.

    **Observations:**
    -	**Consistent Improvement:** Both training and validation accuracy improved steadily with each epoch.
    -	**Decreasing Loss:** Loss values for both training and validation consistently declined, reflecting the model’s increasing ability to correctly classify plant diseases.
    -	**Validation Performance:** The validation accuracy reaching 97.46% highlights the model's robustness on unseen data.
    -	**Consistent Performance:** Validation metrics closely mirrored training metrics, signifying robust model performance on unseen data.

                 
    ##### Performance Metrics
    To evaluate the model, accuracy and loss were used as the primary metrics:
    -	**Accuracy:** Measures the proportion of correctly classified samples.
    -	**Loss:** Represents the model’s prediction error, calculated using Categorical Cross-Entropy Loss.
                 
    The loss function used is: L(y,y^) = - Σci=1 yilog(y^i)
    Where:
    -	L(y,y^) is the categorical cross-entropy loss.
    -	yi is the true label (0 or 1 for each class) from one-hot encoded target vector.
    -	y^I is the predicted probability for class i.
    -	c is the number of classes.

                 
    ##### Test Results
    The trained model was evaluated on the test dataset, which consisted of unseen images of plant leaves. The performance metrics achieved during testing were:
    -	**Test Accuracy:** 97.45%
    -	**Test Loss:** 0.0881

                   
    ##### Model evaluation:
                 
    A confusion matrix was generated to evaluate the model’s performance across different classes. The results indicate high precision and recall, with minimal misclassifications across categories.
                 
    - Class	Precision	Recall	F1-score	Support
    - Apple___Apple_scab	0.94	0.97	0.95	504
    - Apple___Black_rot	0.97	0.99	0.98	497
    - Apple___Cedar_apple_rust	0.95	0.99	0.97	440
    - Apple___healthy	0.97	0.97	0.97	502
    - Blueberry___healthy	0.96	0.98	0.97	454
    - Cherry_(including_sour)___Powdery_mildew	0.99	0.99	0.99	421
    - Cherry_(including_sour)___healthy	0.98	0.99	0.99	456
    - Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot	0.95	0.89	0.92	410
    - Corn_(maize)___Common_rust_	1.00	0.99	0.99	477
    - Corn_(maize)___Northern_Leaf_Blight	0.92	0.97	0.95	477
    - Corn_(maize)___healthy	1.00	1.00	1.00	465
    - Grape___Black_rot	0.98	0.98	0.98	472
    - Grape___Esca_(Black_Measles)	0.99	0.98	0.98	480
    - Grape___Leaf_blight_(Isariopsis_Leaf_Spot)	1.00	1.00	1.00	430
    - Grape___healthy 	1.00	0.99	0.99	423
    - Orange___Haunglongbing_(Citrus_greening)	0.99	0.99	0.99	503
    - Peach___Bacterial_spot	0.96	0.97	0.97	459
    - Peach___healthy	0.98	0.99	0.99	432
    - Pepper,_bell___Bacterial_spot	0.97	0.96	0.97	478
    - Pepper,_bell___healthy	0.99	0.93	0.96	497
    - Potato___Early_blight	0.99	0.98	0.99	485
    - Potato___Late_blight	0.97	0.97	0.97	485
    - Potato___healthy	0.98	0.96	0.97	456
    - Raspberry___healthy	0.99	0.97	0.98	445
    - Soybean___healthy	0.99	0.99	0.99	505
    - Squash___Powdery_mildew	0.97	0.99	0.98	434
    - Strawberry___Leaf_scorch	0.99	0.97	0.98	444
    - Strawberry___healthy	0.95	1.00	0.97	456
    - Tomato___Bacterial_spot	0.96	0.98	0.97	425
    - Tomato___Early_blight	0.95	0.93	0.94	480
    - Tomato___Late_blight	0.96	0.90	0.93	463
    - Tomato___Leaf_Mold	0.97	0.98	0.97	470
    - Tomato___Septoria_leaf_spot	0.96	0.91	0.93	436
    - Tomato___Spider_mites Two-spotted_spider_mite	0.96	0.94	0.95	435
    - Tomato___Target_Spot	0.94	0.93	0.93	457
    - Tomato___Tomato_Yellow_Leaf_Curl_Virus	0.99	1.00	0.99	490
    - Tomato___Tomato_mosaic_virus	0.96	1.00	0.98	448
    - Tomato___healthy	0.99	0.99	0.99	481
                    
    - Accuracy			0.97	17572
    - Macro Average	0.97	0.97	17572
    - Weighted Average	0.97	0.97	17572


    ##### Feature Analysis
    The application of the model in a real-world scenario was tested through its primary features:
                 
    **Image Classification:**
    -	**Result:** The model reliably classified images of plant leaves into their respective categories with high accuracy.
    -	**Discussion:** Users reported satisfaction with the model’s speed and accuracy in predicting plant diseases. However, in cases of poor-quality images (e.g., blurred or poorly lit), the performance slightly declined. Future versions could include preprocessing steps, such as auto-enhancement of input images, to mitigate this issue.
    """)

#Prediction page
elif (app_mode == "Disease Detection"):
    st.header("Disease Detection")
    st.markdown ("""
    ### Chose an Image:
    """)
    test_image = st.file_uploader("")
    if (test_image):
        with st.spinner ("Please Wait...."):
            st.image(test_image, use_column_width = True)

        #prediction image
        if (st.button("Predict")):
            with st.spinner ("Please Wait...."):
                result_index = model_prediction(test_image)

            #define class
            class_name = ['Apple_scab',
            'Apple___Black_rot',
            'Apple___Cedar_apple_rust',
            'Apple___healthy',
            'Blueberry___healthy',
            'Cherry_(including_sour)___Powdery_mildew',
            'Cherry_(including_sour)___healthy',
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
            'Corn_(maize)___Common_rust_',
            'Corn_(maize)___Northern_Leaf_Blight',
            'Corn_(maize)___healthy',
            'Grape___Black_rot',
            'Grape___Esca_(Black_Measles)',
            'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
            'Grape___healthy',
            'Orange___Haunglongbing_(Citrus_greening)',
            'Peach___Bacterial_spot',
            'Peach___healthy',
            'Pepper,_bell___Bacterial_spot',
            'Pepper,_bell___healthy',
            'Potato___Early_blight',
            'Potato___Late_blight',
            'Potato___healthy',
            'Raspberry___healthy',
            'Soybean___healthy',
            'Squash___Powdery_mildew',
            'Strawberry___Leaf_scorch',
            'Strawberry___healthy',
            'Tomato___Bacterial_spot',
            'Tomato___Early_blight',
            'Tomato___Late_blight',
            'Tomato___Leaf_Mold',
            'Tomato___Septoria_leaf_spot',
            'Tomato___Spider_mites Two-spotted_spider_mite',
            'Tomato___Target_Spot',
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
            'Tomato___Tomato_mosaic_virus',
            'Tomato___healthy']

            st.write("Our Prediction")
            st.success("The model is Predicting it's {}". format(class_name[result_index]))