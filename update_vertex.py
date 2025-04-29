import pandas as pd
from surprise import Dataset, Reader, SVD
from google.cloud import aiplatform, storage # Added GCS/Vertex clients
import pickle
import os
import time
import argparse
from typing import List, Dict, Tuple, Optional # Added Optional

# --- Firebase Admin Setup (REMOVED - no longer fetching directly) ---
# import firebase_admin
# from firebase_admin import credentials, firestore
# if not firebase_admin._apps:
#     firebase_admin.initialize_app()
# db = firestore.client()
# --- End Firebase Admin Setup ---

class ModelUpdaterVertex:
    # REMOVED load_or_create_model - Vertex jobs start fresh, we always create/train
    def __init__(self, model_output_dir: str, project_id: str, region: str, model_display_name: str):
        """Initialize the model updater for Vertex AI Training."""
        self.model_output_dir = model_output_dir # GCS path provided by Vertex AI
        self.project_id = project_id
        self.region = region
        self.model_display_name = model_display_name
        self.model: SVD = SVD() # Always create a new SVD instance for training
        self.user_interactions_df: Optional[pd.DataFrame] = None
        # Define local path WITHIN the container for temporary model saving
        self.local_model_filename = 'trained_svd_model.pkl'

    def load_data_from_gcs(self, gcs_data_path: str) -> bool:
        """Load interaction data from CSV file(s) in GCS."""
        print(f"Loading interaction data from GCS path: {gcs_data_path}")
        try:
            # Assuming data is exported as CSV(s) with columns: user_id, outfit_id, interaction
            # Adapt glob pattern if needed (e.g., '/*.csv' for multiple files)
            self.user_interactions_df = pd.read_csv(f"{gcs_data_path}/interactions.csv") # Adjust filename/pattern

            if self.user_interactions_df.empty:
                print("Warning: Loaded DataFrame is empty.")
                return False

            # Basic validation (optional but recommended)
            required_cols = ['user_id', 'outfit_id', 'interaction']
            if not all(col in self.user_interactions_df.columns for col in required_cols):
                 raise ValueError(f"Input CSV must contain columns: {required_cols}")

            print(f"Loaded DataFrame with {len(self.user_interactions_df)} interactions from GCS.")
            return True

        except FileNotFoundError:
             print(f"ERROR: No file found at {gcs_data_path}/interactions.csv")
             return False
        except Exception as e:
            print(f"ERROR loading data from GCS: {e}")
            import traceback
            traceback.print_exc()
            return False

    # Renamed from fetch_all_interactions
    # def fetch_all_interactions(self) -> bool: # REMOVED

    def train_and_save_model_local(self) -> bool:
        """Train the SVD model and save it locally within the container."""
        if self.user_interactions_df is None or self.user_interactions_df.empty:
            print("No interaction data loaded, skipping model training.")
            return False

        try:
            print("Preparing data for Surprise...")
            # Convert IDs to string just in case they are loaded as numbers
            self.user_interactions_df['user_id'] = self.user_interactions_df['user_id'].astype(str)
            self.user_interactions_df['outfit_id'] = self.user_interactions_df['outfit_id'].astype(str)

            min_rating = self.user_interactions_df['interaction'].min()
            max_rating = self.user_interactions_df['interaction'].max()
            print(f"Detected rating scale: min={min_rating}, max={max_rating}")
            if pd.isna(min_rating) or pd.isna(max_rating):
                 raise ValueError("Cannot determine rating scale. Check data.")

            reader = Reader(rating_scale=(min_rating, max_rating))
            data = Dataset.load_from_df(
                self.user_interactions_df[['user_id', 'outfit_id', 'interaction']],
                reader
            )

            print("Building trainset...")
            trainset = data.build_full_trainset()
            print(f"Trainset built with {trainset.n_users} users and {trainset.n_items} items.")

            print("Training SVD model...")
            self.model.fit(trainset) # Train the SVD model instance

            print(f"Saving trained model locally to {self.local_model_filename}...")
            with open(self.local_model_filename, 'wb') as f:
                pickle.dump(self.model, f)

            print("Model trained and saved locally successfully!")
            return True

        except Exception as e:
            print(f"ERROR during model training/local saving: {e}")
            import traceback
            traceback.print_exc()
            return False

    def upload_model_to_gcs(self) -> Optional[str]:
        """Uploads the locally saved model file to the GCS output directory."""
        if not os.path.exists(self.local_model_filename):
             print(f"Local model file {self.local_model_filename} not found. Cannot upload.")
             return None

        try:
            storage_client = storage.Client()
            # model_output_dir looks like gs://bucket-name/path/to/output/
            # Extract bucket name and blob path
            if not self.model_output_dir.startswith("gs://"):
                raise ValueError("model_output_dir must be a GCS path (gs://...).")

            bucket_name = self.model_output_dir.split('/')[2]
            # Path within the bucket, removing gs://<bucket_name>/ and ensuring trailing /
            blob_prefix = '/'.join(self.model_output_dir.split('/')[3:])
            if blob_prefix and not blob_prefix.endswith('/'):
                blob_prefix += '/'
            destination_blob_name = f"{blob_prefix}{self.local_model_filename}"

            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(destination_blob_name)

            print(f"Uploading {self.local_model_filename} to {self.model_output_dir}{self.local_model_filename}...")
            blob.upload_from_filename(self.local_model_filename)
            print("Model uploaded to GCS successfully.")
            # Return the GCS directory containing the model
            return self.model_output_dir

        except Exception as e:
            print(f"Error uploading model to GCS: {e}")
            return None

    def register_model_in_vertexai(self, model_gcs_uri: str):
        """Registers the model from GCS into Vertex AI Model Registry."""
        if not model_gcs_uri:
            print("Model GCS URI not provided, skipping registration.")
            return

        try:
            print(f"Registering model from {model_gcs_uri} in Vertex AI...")
            aiplatform.init(project=self.project_id, location=self.region)

            # Define the serving container - use a pre-built scikit-learn container
            # Check Vertex AI docs for the latest compatible scikit-learn container image URI
            # Make sure the version matches the one 'surprise' might implicitly depend on or is compatible
            serving_image = 'us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest' # Example, verify latest/correct one

            model = aiplatform.Model.upload(
                display_name=self.model_display_name,
                artifact_uri=model_gcs_uri, # GCS *directory* where model is saved
                serving_container_image_uri=serving_image,
                # serving_container_predict_route="/predict", # Defaults are usually fine for pre-built
                # serving_container_health_route="/healthz",
                description='SVD Collaborative Filtering model for Aura app (trained via custom job)'
            )
            print(f"Successfully registered model: {model.resource_name}")

        except Exception as e:
            print(f"ERROR registering model in Vertex AI: {e}")

# --- Main Execution Logic for Vertex AI Job ---
def main():
    parser = argparse.ArgumentParser(description='Train SVD recommender model using data from GCS and register to Vertex AI')
    # Arguments provided by Vertex AI Custom Job + User defined
    parser.add_argument('--data-path', required=True, type=str, help='GCS path to the directory containing interaction data (e.g., gs://bucket/data/)')
    parser.add_argument('--model-output-dir', required=True, type=str, help='GCS directory path to save the trained model artifact (e.g., gs://bucket/models/run_timestamp/)')
    parser.add_argument('--project-id', required=True, type=str, help='Google Cloud Project ID')
    parser.add_argument('--region', required=True, type=str, help='Google Cloud Region (e.g., us-central1)')
    parser.add_argument('--model-display-name', type=str, default='aura-svd-recommender', help='Display name for the model in Vertex AI Registry')
    # REMOVED --interval argument

    args = parser.parse_args()
    print(f"Starting training job with args: {args}")

    updater = ModelUpdaterVertex(
        model_output_dir=args.model_output_dir,
        project_id=args.project_id,
        region=args.region,
        model_display_name=args.model_display_name
    )

    # 1. Load data
    if not updater.load_data_from_gcs(args.data_path):
        print("Failed to load data. Exiting.")
        exit(1) # Signal failure to Vertex AI

    # 2. Train and save model locally in the container
    if not updater.train_and_save_model_local():
        print("Failed to train/save model locally. Exiting.")
        exit(1)

    # 3. Upload model from local container path to GCS output dir
    model_gcs_uri = updater.upload_model_to_gcs()
    if not model_gcs_uri:
        print("Failed to upload model to GCS. Exiting.")
        exit(1)

    # 4. Register model from GCS to Vertex AI Model Registry
    updater.register_model_in_vertexai(model_gcs_uri)

    print("Training job finished successfully.")

if __name__ == "__main__":
    main()