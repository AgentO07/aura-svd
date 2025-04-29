# Use a standard Python 3.9 image as the base
FROM python:3.9-slim

# Set a working directory inside the container image
WORKDIR /app

# *** ADD THIS SECTION TO INSTALL BUILD TOOLS ***
# Update package lists and install build-essential (contains gcc)
# --no-install-recommends keeps the image smaller
# Clean up apt cache afterwards to reduce image size
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*
# *** END OF ADDED SECTION ***

# Copy ONLY the requirements file first (for better caching)
COPY requirements.txt .

# Install all libraries listed in requirements.txt
# (Optional: keep the pip upgrade line if you added it)
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code (your training script)
COPY update_vertex.py .

# Set the command that will run when the container starts
ENTRYPOINT ["python", "update_vertex.py"]
