# 📦 Question Answering for Compressor Manual

Description: "Question Answering for Compressor Manual" is a Streamlit-based web application that provides instant answers to user queries regarding FAQs in a Renner Compressor manual. Utilizing a Retrieval-Augmented Generation (RAG) system, the app quickly identifies relevant information and synthesizes precise answers. An optional reranker can further refine the accuracy of responses. For more in-depth information, users can access the full [Renner Compressor Manual (PDF)](https://renner-rus.ru/upload/iblock/881/8813b4f3ce3b95c2386344cb55453e44.pdf).


## Demo App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://qa-streamlit-test-c44c0e823dd8.herokuapp.com/)

## Using the App

Here's a screenshot of the app in action:

![App Screenshot](https://imgur.com/a/bxhlwGl)

The "Question Answering for Compressor Manual" app is designed to be user-friendly and intuitive. Follow the steps below to interact with the app and get answers to your questions:

### Step 1: Select LLM Model

Upon launching the app, you will be presented with an option to select the Language Model (LLM) you wish to use for answering questions. Choose from the following options:

- **Mixtral 8x7B model**: A powerful language model that provides comprehensive answers.
- **Mistral 7B Instruct v01**: A model fine-tuned to follow instructions and provide more precise answers.

These models are powered by the Anyscale API.

### Step 2: Select Retrieval Embedding Model

Next, select the Retrieval Embedding Model that will be used to retrieve relevant information from the compressor manual. You have two options:

- **Default "text-embedding-ada-002"**: An embedding model from OpenAI that captures the semantic meaning of text.
- **Local "BAAI/bge-small-en"**: A smaller, efficient model hosted locally via Huggingface, suitable for quick retrieval tasks.

### Step 3: Cohere Reranker Option

If you wish to refine the search results further, you can opt to use the Cohere Reranker. Simply click to enable or disable this option based on your preference.

### Step 4: Enter Your Question

In the provided text field, type in the question you want to ask about the Renner Compressor manual. Be as specific as possible to get the best results.

### Step 5: Submit Your Query

After filling in all the required information, click the "Submit" button to send your query to the system. The app will process your question using the selected models and return an answer to you.

Please be patient as the system retrieves and processes the information, as this may take a few moments depending on the complexity of the question and the models selected.

### Interacting with the Results

Once the answer is displayed, you can:

- Read through the provided response to find the information you were looking for.
- If needed, refine your question or select different models to get a different perspective on the answer.

The app is designed to provide quick responses to help you understand the intricacies of the Renner Compressor manual without the need to search through the document manually.

## How to build

1. Clone the repository

https://github.com/finoceva/qa-streamlit-test.git

2. Navigate to the project directory:

cd qa-streamlit-test

3. Install the required dependencies:

pip install -r requirements.txt

4. Run the Streamlit app:

streamlit run app.py

The app will start and be available in your web browser at http://localhost:8501.

## Dockerizing the App Locally

To run the "Question Answering for Compressor Manual" app within a Docker container, follow these simple steps:

### Prerequisites

- Have the `Dockerfile` and `.env` file at the root of your project directory.

### Build the Docker Image

Build your Docker image by running the following command in your terminal, replacing `qa-streamlit-test` with your preferred image name:

```sh
docker build -t qa-streamlit-test .
```
### Run the Docker Container
Start the container with the following command, which mounts your .env file and maps the local port to the container's Streamlit port:
```
docker run -d -p 8501:8501 -e PORT=8501 -v $(pwd)/.env:/app/.env qa-streamlit-test
```
### Access the App
Open a web browser and navigate to http://localhost:8501 to use the app.

## Cloud Deployment
The "Question Answering for Compressor Manual" app is deployed on Heroku, a cloud platform that enables developers to build, run, and operate applications entirely in the cloud.

### How the App is Deployed to Heroku

The app is containerized using Docker, which ensures that it runs consistently across different environments. The Docker container includes all the necessary dependencies specified in the `requirements.txt` file.

Here's an overview of the deployment process:

1. **Dockerization**: The app is packaged into a Docker container using a `Dockerfile` that specifies the base image, dependencies, and commands needed to run the app.

2. **Heroku Setup**: A new app is created on Heroku, and the Heroku Container Registry is used to store the Docker image.

3. **Deployment**: The Docker image is pushed to the Heroku Container Registry, and then a release is created to deploy the app.

4. **Accessing the App**: Once deployed, the app can be accessed via a Heroku-provided URL, which is publicly available.

### Steps to Deploy the App

To deploy the app to Heroku, follow these steps:

1. Install the [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli) and log in to your Heroku account.

   ```sh
   heroku login

2. Navigate to the project directory and log in to the Heroku Container Registry.
    ```sh
   heroku container:login

3. Create a new Heroku app.
   ```sh
   heroku create <your-app-name>

4. Build the Docker image and push it to the Heroku Container Registry.
    ```sh
    heroku container:push web -a <your-app-name>

5. Release the image to deploy the app.
    ```sh
    heroku container:release web -a <your-app-name>

6. Open the app in a web browser.
    ```sh
    heroku open -a <your-app-name>

## Configuration
The app is configured to use environment variables for sensitive information, such as API keys. These variables are set in the Heroku app settings under "Config Vars."
For local development, environment variables can be set in a .env file, which is not tracked by version control for security reasons.

### Environment Configuration

The application requires a set of environment variables to be set in order to interact with various APIs and services. These variables should be defined in a `.env` file located at the root of the project directory. This file is not included in version control for security reasons, as it contains sensitive information such as API keys.

Here is the structure of the `.env` file required for the application:

```plaintext
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_API_BASE=your_openai_api_base_url_here
OPENAI_API_TYPE=your_openai_api_type_here

# Anyscale Configuration
ANYSCALE_API_BASE=your_anyscale_api_base_url_here
ANYSCALE_API_KEY=your_anyscale_api_key_here

# Google Cloud Configuration
GOOGLE_API_KEY=your_google_cloud_api_key_here

# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=your_pinecone_environment_here

# Cohere Configuration
COHERE_API_KEY=your_cohere_api_key_here
```
