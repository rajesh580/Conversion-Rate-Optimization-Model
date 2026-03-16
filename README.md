# Interactive Customer Conversion Prediction Web App

This is a Flask-based web application that uses machine learning to predict whether an online shopper will complete a purchase. It features a highly interactive front-end where users can input session data and instantly see predictions from multiple models.

A unique feature of this project is the client-side implementation of the trained models in JavaScript, allowing for real-time predictions and feature-contribution analysis without needing to call the backend for every input change.

![Project Screenshot](https://i.imgur.com/sK8aH8r.png)

## 🚀 Features

-   **Real-time Prediction:** Get instant conversion predictions as you adjust 17 different user session features.
-   **Multiple Models:** Compare predictions from three different ML models (Decision Tree, Logistic Regression, KNN) and an ensemble model.
-   **Client-Side ML:** The core prediction logic is implemented directly in JavaScript, providing a fast and responsive user experience. The models (Decision Tree rules, Logistic Regression weights, and a KNN approximation) are hardcoded into the frontend for immediate feedback.
-   **Feature Contribution Analysis:** For Logistic Regression, the app visualizes how each input feature contributes to the final prediction, offering insights into the model's decision-making process.
-   **Interactive Visualization:** A dynamic chart shows how the predicted conversion rate changes with the number of product page visits.
-   **Responsive Design & Theme Toggle:** A modern, clean UI with light and dark modes that works across different devices.

## 🛠️ Tech Stack

-   **Backend:** Python, Flask
-   **Machine Learning:** Scikit-learn, Pandas, NumPy
-   **Frontend:** HTML5, CSS3, JavaScript (ES6)
-   **Charting:** Chart.js

## 📂 Project Structure

```
.
├── app.py                      # Flask backend, data preprocessing, and model training
├── online_shoppers_intention.csv # The dataset
├── templates/
│   └── index.html              # The single-page frontend with HTML, CSS, and JS
└── README.md                   # This file
```

## 📊 Dataset

The project uses the **"Online Shoppers Purchasing Intention Dataset"** from the UCI Machine Learning Repository. This dataset consists of feature vectors belonging to 12,330 sessions.

-   **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset)
-   **Citation:** Sakar, C.O., Polat, S.O., Katircioglu, M. et al. Real-time prediction of online shoppers’ purchasing intention using multilayer perceptron and LSTM recurrent neural networks. Neural Comput & Applic (2019) 31: 6765.

## 🤖 Models

The application trains and utilizes four different prediction models. The backend (`app.py`) trains the models using Scikit-learn on the full dataset upon starting. The key parameters and logic of these trained models are then implemented in the frontend's JavaScript (`templates/index.html`) to perform live predictions directly in the browser.

1.  **Decision Tree:** A simple and interpretable model. The tree's rules (up to a max depth of 5) are implemented in JavaScript for client-side prediction.
2.  **Logistic Regression:** A linear model for binary classification. The trained coefficients and intercept are used in the frontend to calculate feature contributions and prediction probability.
3.  **K-Nearest Neighbors (KNN):** A non-parametric method. The frontend uses a cluster-based approximation with 8 representative cluster centers for fast client-side execution, simulating a k=7 model.
4.  **Ensemble:** A simple soft-voting ensemble that averages the probabilities of the three base models.

## ⚙️ Setup and Installation

To run this project locally, follow these steps:

1.  **Clone the repository**
    ```bash
    git clone https://github.com/rajesh580/Conversion-Rate-Optimization-Model.git
    cd Conversion-Rate-Optimization-Model
    ```

2.  **Create and activate a virtual environment**
    ```bash
    # Create the virtual environment
    python -m venv venv

    # Activate on Windows
    .\venv\Scripts\activate

    # Activate on macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies**

    Create a `requirements.txt` file with the following content:
    ```txt
    Flask
    pandas
    scikit-learn
    numpy
    ```
    Then install the packages:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the dataset**

    Download the dataset from the UCI repository and place the `online_shoppers_intention.csv` file in the root directory of the project.

5.  **Run the Flask application**
    ```bash
    python app.py
    ```

6.  **View the application**

    Open your web browser and navigate to `http://127.0.0.1:5000`.

## 💡 Usage

-   Open the web application in your browser.
-   The left panel contains input fields for all 17 features of a user session.
-   Adjust the sliders, dropdowns, and number inputs to simulate different user behaviors. The prediction results on the right panel update in real-time.
-   Choose a specific ML algorithm (Decision Tree, Logistic Regression, KNN, or Ensemble) to see its prediction.
-   The "LR Feature Contributions" card shows a detailed breakdown of how the Logistic Regression model arrived at its decision.
-   The chart at the bottom visualizes how conversion probability changes based on the number of product-related pages visited.