<h1 align="center">Geospatial Data Web App</h1>

<p align="center">
  <i>A Geospatial Data Visualization and Land Cover Prediction web application developed in Flask</i>
</p>


<br>
<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white" alt="Numpy" />
  <img src="https://img.shields.io/badge/Jupyter-F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white" alt="Jupyter" />
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow" />
  <img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white" alt="Keras" />
  <img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask" />
  <img src="https://img.shields.io/badge/GeoPandas-2F3D4E?style=for-the-badge&logo=geopandas&logoColor=white" alt="GeoPandas" />
  <img src="https://img.shields.io/badge/HTML-E34F26?style=for-the-badge&logo=html5&logoColor=white" alt="HTML" />
  <img src="https://img.shields.io/badge/CSS-1572B6?style=for-the-badge&logo=css3&logoColor=white" alt="CSS" />
  <img src="https://img.shields.io/badge/JavaScript-F7E018?style=for-the-badge&logo=javascript&logoColor=white" alt="JavaScript" />
  <img src="https://img.shields.io/badge/Ajax-003A52?style=for-the-badge&logo=ajax&logoColor=white" alt="Ajax" />
  <img src="https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white" alt="Bootstrap" />
  <img src="https://img.shields.io/badge/Amazon_AWS-232F3E?style=for-the-badge&logo=amazon-aws&logoColor=white" alt="Amazon AWS" />
  <img src="https://img.shields.io/badge/EC2-FF9900?style=for-the-badge&logo=amazon-aws&logoColor=white" alt="EC2" />
  <img src="https://img.shields.io/badge/Lambda-FF9900?style=for-the-badge&logo=amazon-aws&logoColor=white" alt="Lambda" />
</div>


# 📂 Project Overview

<p align="center">
  <a href="/Machine%20Learning%20-%20Training%20and%20Testing/ML_Training_and_Testing_EuroSAT_ResNet50.ipynb">
    <img src="https://img.shields.io/badge/Jupyter%20Notebook-Model%20Training%20%26%20Performance%20Testing-ff69b4?style=for-the-badge&logo=jupyter" alt="Model Training & Performance Testing">
  </a>
<a href="/Machine%20Learning%20-%20Training%20and%20Testing/Testing_Model_Single_Image.ipynb">
    <img src="https://img.shields.io/badge/Jupyter%20Notebook-Testing%20on%20Single%20Images-ff69b4?style=for-the-badge&logo=jupyter" alt="Testing on Single Images">
</a>

</p>


- [**Model Training & Performance Testing**](/Machine%20Learning%20-%20Training%20and%20Testing/ML_Training_and_Testing_EuroSAT_ResNet50.ipynb): This Jupyter notebook goes into detail of training the ResNet-50 model on the EUROSAT RGB dataset.

- [**Testing on Single Images**](/Machine%20Learning%20-%20Training%20and%20Testing/Testing_Model_Single_Image.ipynb): This Jupyter notebook explores the usage of the model on single images, as this will be the real-life use case when it is embedded in the web application.


### Web App

This directory contains the code for web application. The application is deployed in flask with dynamic content loading therefore a single html file along with it. There is also a folder that contains the lambda function code.


# Documentation

## Web Application

### Features

- **Upload and Visualize Geospatial Data**: Supports GeoJSON and shapefiles (inside ZIP archives).
- **Predict Land Cover**: Uses a pre-trained ResNet50 model to classify land cover from uploaded images.
- **User-friendly Interface**: Built with Bootstrap and Leaflet for interactive data visualization.

### Technologies Used

- **Flask**: Web framework for Python.
- **TensorFlow**: Machine learning library for prediction.
- **Boto3**: AWS SDK for Python to interact with AWS Lambda.
- **GeoPandas**: For reading and handling geospatial data.
- **OpenCV**: For image processing.

### Installation

1. **Clone the Repository**

    ```bash
    git clone https://github.com/yourusername/geospatial-data-app.git
    cd geospatial-data-app
    ```

2. **Create a Virtual Environment**

    ```bash
    python -m venv venv
    ```

3. **Activate the Virtual Environment**

    - On Windows:

        ```bash
        venv\Scripts\activate
        ```

    - On macOS/Linux:

        ```bash
        source venv/bin/activate
        ```

4. **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

5. **Download the Pre-trained Model**

    Download the `ResNet50_eurosat.h5` model and place it in the `models/` directory.

6. **Configure AWS Lambda**

    Set up an AWS Lambda function named `ml_preprocessing` and ensure it has the appropriate permissions to handle image processing. Update the `lambda_client` configuration in `app.py` with your AWS region and credentials if needed.

### Running the Application

1. **Start the Flask Application**

    ```bash
    python app.py
    ```

2. **Access the Web Application**

    Open your browser and navigate to `http://localhost:80` to view the application.

## API Endpoints

### `/`

Displays the home page with options to visualize data and predict land cover.

### `/upload`

**Method**: POST

**Description**: Uploads a file (GeoJSON or ZIP containing shapefiles) and returns the GeoJSON representation of the data.

**Parameters**:

- `file`: The file to upload.

**Response**:

- `geojson`: GeoJSON representation of the uploaded data.
- `error`: Error message if the file is not valid.

### `/predict_image`

**Method**: POST

**Description**: Uploads an image for land cover prediction.

**Parameters**:

- `image`: The image file to upload.

**Response**:

- `prediction`: The predicted land cover class.
- `error`: Error message if the file is not valid.

### HTML/CSS Components

- **Header**: Fixed header with the application title.
- **Sidebar**: Navigation links for home, visualizing data, and predicting land cover.
- **Main Content**: Dynamic content area updated based on user interaction.
- **Footer**: Fixed footer with developer information.

**Sample HTML and CSS**:

See `templates/index.html` for the HTML structure and `static/styles.css` for custom styles.

## Artificial Intelligence

This AI part of this project includes the process of training and evaluating a ResNet50 model on the EUROSAT dataset. It includes data preprocessing, data augmentation, model training, and performance evaluation. The following sections detail each step involved in the process.

### 1. Importing Libraries

To start, the necessary libraries are imported. These include libraries for file operations, data manipulation, image processing, data visualization, machine learning, and model training. The key libraries used are:

- **os** and **shutil**: For interacting with the operating system and managing files.
- **random** and **tqdm**: For generating random numbers and displaying progress bars.
- **numpy** and **pandas**: For numerical operations and data manipulation.
- **PIL**: For image processing.
- **seaborn** and **matplotlib.pyplot**: For creating visualizations.
- **sklearn.model_selection.StratifiedShuffleSplit**: For splitting data into training and testing sets.
- **keras.preprocessing.image.ImageDataGenerator**: For data augmentation.
- **keras.optimizers.Adam**: For optimizing the model.
- **keras.callbacks.ModelCheckpoint, EarlyStopping, ReduceLROnPlateau**: For managing model training and checkpoints.
- **keras.applications.ResNet50**: For the ResNet50 architecture.
- **keras.models.Model, keras.layers.Dense, keras.layers.GlobalAveragePooling2D**: For building and customizing the model.

### 2. Dataset and Class Distribution

#### Loading Dataset

The dataset directory is specified, and class labels are extracted from the dataset. This allows for initial inspection of the dataset's structure.

#### Plotting Class Distribution

A bar chart is plotted to visualize the distribution of images across different classes in the dataset. This helps in understanding the balance of the dataset.

### 3. Data Preparation

#### Creating Directories for Training and Testing

Directories for training and testing data are created if they do not already exist. These directories will store the images for model training and evaluation.

#### Splitting and Moving Images

The dataset is split into training and testing sets using `StratifiedShuffleSplit` to maintain class distribution. Images are moved to the appropriate directories for training and testing.

### 4. Data Augmentation and Generators

#### Creating Data Generators

Data generators are set up for both training and testing data. The training data is augmented with various transformations such as rotation, width/height shifts, shear, zoom, and flips. The testing data is rescaled but not augmented.

### 5. Model Training

#### Model Definition and Compilation

A function is defined to create and compile the ResNet50 model. This function allows for optional fine-tuning of the model layers. The model is built using ResNet50 with pre-trained weights and customized with a global average pooling layer and a dense output layer.

#### Training the Model

The model is trained using the training data generator and evaluated on the testing data generator. Callbacks such as ModelCheckpoint, EarlyStopping, and ReduceLROnPlateau are used to manage the training process and optimize performance.

#### Evaluating the Model

The best model weights are loaded, and the model's performance is evaluated using the test data. Predictions are generated, and performance metrics such as precision, recall, and F1 score are computed.

### 6. Performance Results

The final model achieves an accuracy of 0.9676 with a global F2 score of 0.9676. Detailed performance metrics for each class, including precision, recall, and F1 score, are provided to assess the model's effectiveness.

The following table summarizes the performance metrics of the ResNet50 model on the EUROSAT dataset.

| Class                | Precision | Recall   | F-Score  | Support |
|----------------------|-----------|----------|----------|---------|
| AnnualCrop           | 0.9633    | 0.9602   | 0.9617   | 628     |
| Forest               | 0.9481    | 1.0000   | 0.9734   | 603     |
| HerbaceousVegetation | 0.9926    | 0.9210   | 0.9554   | 582     |
| Highway              | 0.9790    | 0.9554   | 0.9671   | 538     |
| Industrial           | 0.9710    | 0.9533   | 0.9621   | 492     |
| Pasture              | 0.9836    | 0.9497   | 0.9664   | 378     |
| PermanentCrop        | 0.9297    | 0.9646   | 0.9468   | 480     |
| Residential          | 0.9565    | 0.9984   | 0.9770   | 639     |
| River                | 0.9646    | 0.9800   | 0.9723   | 501     |
| SeaLake              | 0.9964    | 0.9821   | 0.9892   | 559     |

---

This documentation provides a comprehensive overview of the AI module, including data preparation, model training, and evaluation. For further details or questions, please refer to the specific sections or contact the maintainer.

## AWS Integration Documentation

This section documents the integration of a Flask application deployed on an AWS EC2 instance, utilizing AWS Lambda for efficient processing of land cover predictions from satellite imagery.

### Architecture Overview

The Flask app is hosted on an EC2 instance, while resource-intensive tasks like image processing and land cover prediction are offloaded to an AWS Lambda function. The system supports various geospatial formats and was tested thoroughly for scalability and performance.

---

### EC2 Instance Setup

1. **Instance Configuration**:
   - Created an EC2 instance with Ubuntu for hosting the Flask application.
   - Increased EC2 storage to **15 GB** using the following commands:
     ```bash
     sudo growpart /dev/nvme0n1 1
     sudo resize2fs /dev/nvme0n1p1
     ```
   - Created a **5 GB swap file** to manage memory more efficiently:
     ```bash
     sudo dd if=/dev/zero of=/swapfile bs=1M count=5120
     sudo chmod 600 /swapfile
     sudo mkswap /swapfile
     sudo swapon /swapfile
     ```
   - Added swap file entry to `/etc/fstab` to ensure it persists across reboots:
     ```bash
     /swapfile swap swap defaults 0 0
     ```

---

### Flask Application Deployment

1. **Web Server Setup**:
   - Deployed the Flask app using **NGINX** and **Gunicorn**.
   - Gunicorn is responsible for handling requests to the Flask app, while NGINX serves as a reverse proxy to handle incoming HTTP traffic.
   - Configured NGINX to serve the app on **port 80** for public access, allowing HTTP access via the public IP or domain.

2. **NGINX Configuration**:
   - Configuration for NGINX to proxy requests to Gunicorn:
     ```nginx
     server {
         listen 80;
         server_name your_public_ip;

         location / {
             proxy_pass http://127.0.0.1:5000;
             proxy_set_header Host $host;
             proxy_set_header X-Real-IP $remote_addr;
             proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
             proxy_set_header X-Forwarded-Proto $scheme;
         }
     }
     ```

3. **Gunicorn Configuration**:
   - Start the Flask application using Gunicorn with:
     ```bash
     gunicorn --workers 3 --timeout 120 --bind 0.0.0.0:5000 app:app
     ```
   - The application is served on port 5000 and proxied through NGINX. It can be accessed on port 80, by simply sending an HTTP request.

---

### AWS Lambda Setup for Model Processing

1. **Lambda Function**:
   - Created a Lambda function to perform **land cover prediction** using machine learning models.
   - Created a **Lambda Layer** containing **NumPy**, **TensorFlow**, and **Keras**, ensuring the model dependencies are efficiently managed:
     - Packaged the dependencies in a `.zip` file and uploaded them to an S3 bucket.
     - Attached the Lambda Layer to the function for use in predictions.
   
2. **S3 Integration**:
   - Uploaded large datasets and models to an **S3 bucket** for use in Lambda.
   - Lambda pulls data from S3, performs processing, and returns results to the Flask application.

3. **EC2-Lambda Integration**:
   - The EC2 instance communicates with the Lambda function using AWS SDK or HTTP requests.
   - Requests from the Flask app trigger the Lambda function to process satellite imagery data, offloading heavy computations to Lambda and improving EC2 efficiency.

---

### Data Visualization

The application supports the following geospatial file formats for visualization and processing:
- **JSON**
- **GeoJSON**
- **Shape Files**
- **GML**
- **KML**
- **TopoJSON**

These formats allow the app to process and visualize a variety of geospatial datasets, ensuring compatibility with different data types and sources.

---

### Testing and Verification

1. **Model Testing**:
   - The application was rigorously tested using a wide range of satellite imagery categories.
   - Ensured that predictions were accurate for multiple land cover types.

2. **Visualization Testing**:
   - Successfully visualized results using the supported geospatial file formats.
   - Tested the app with different file sizes and formats, confirming that it handled them without performance degradation.

---



### Contributing

Feel free to fork the repository and submit pull requests. For major changes or feature requests, please open an issue to discuss them first.



---
