# Balinese Souvenir Classification API

This is a Flask-based API for classifying Balinese souvenirs using a pre-trained TensorFlow model. The API processes uploaded images, predicts the souvenir type, and returns the class label with accuracy.

## Features

- **Image Classification**: Supports 21 categories of Balinese souvenirs.
- **Secure Access**: API protected by a token-based authentication system.
- **Custom ML Model**: Uses a TensorFlow model (`model-surevenir.h5`) trained on souvenir images.
- **Preprocessing Pipeline**: Automatically resizes and normalizes images for predictions.
- **Production-Ready**: Deployable with Docker and Google Cloud Build.

## Project Structure

```
├── app.py             # Main Flask application
├── model/             # Folder containing the machine learning model
├── requirements.txt   # Python dependencies
├── Dockerfile         # Configuration for Docker containerization
├── cloudbuild.yaml    # Google Cloud Build configuration
├── .env               # Environment variables (e.g., SECRET_TOKEN)
└── README.md          # Project documentation
```

## API Endpoints

### `/predict`

**Method**: `POST`

**Description**: Accepts an image and token to predict the souvenir type.

**Request Parameters**:
- `token`: Authorization token (form-data).
- `image`: The uploaded image file (form-data).

**Response**:
- `success`: Indicates whether the prediction was successful.
- `message`: Describes the result.
- `data`: Contains the predicted class and accuracy.

## Getting Started

### Prerequisites

- Python 3.9 or later
- TensorFlow 2.15.0
- Docker (for containerized deployment)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/surevenir/surevenir-predict-api.git
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up your .env file:

```
SECRET_TOKEN=your_secret_token
```

4. Run the application:

```bash
python app.py
```

5. Access the API at `http://localhost:5000`.

### Deployment

1. Build the Docker image:

```bash
docker build -t <tag> .
```

2. Push image:

```bash
docker push <tag>
```

## Acknowledgements

Special thanks to all team members for their hard work and dedication to making this project successful.

## Thank You
