# Use the official Python 3.9 image as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the content of the local directory to the working directory
COPY . .

# Expose the Streamlit port
EXPOSE 8501

# Specify the command to run on container start
CMD streamlit run app.py --server.port $PORT --server.headless true --server.enableCORS false --server.enableXsrfProtection false



# # Use the official Python base image
# FROM python:3.9-slim

# # Set the working directory in the container
# WORKDIR /app

# # Copy the requirements.txt file to the container
# COPY requirements.txt .

# # Install the Python dependencies, including OpenCV and libgl1-mesa-glx
# RUN apt-get update \
#     && apt-get install -y libgl1-mesa-glx \
#     && rm -rf /var/lib/apt/lists/* \
#     && pip install --no-cache-dir -r requirements.txt

# # Copy the source code to the container
# COPY . .

# # Expose the port that Streamlit app will run on
# EXPOSE 8501

# # Set the command to run Streamlit app
# CMD ["streamlit", "run", "app.py"]