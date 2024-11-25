# Step 1: Build stage
FROM python:3.9-slim AS build-stage

WORKDIR /app

# Copy the requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI app code and the model directory
COPY api.py .
COPY plant_disease /app/plant_disease  
COPY class_indices.json /app/class_indices.json
# Step 2: Production stage
FROM python:3.9-slim AS production-stage

WORKDIR /app

# Copy the necessary files from the build-stage
COPY --from=build-stage /app/requirements.txt /app/requirements.txt
COPY --from=build-stage /app/api.py /app/api.py
COPY --from=build-stage /app/plant_disease /app/plant_disease
COPY --from=build-stage /app/class_indices.json /app/class_indices.json 

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the FastAPI port (8000)
EXPOSE 8000

# Command to run the FastAPI app
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
