import pandas as pd
from surprise import Dataset, Reader, SVD
from google.cloud import aiplatform, storage
import pickle
import os
import time
import argparse
from typing import List, Dict, Tuple, Optional

# NO Firebase Admin import needed here

class ModelUpdaterVertexGCS: # Renamed class for clarity
    def __init__(self, model_output_dir: str, project_id: str, region: str, model_display_name: str):
        self.model_output_dir = model_output_dir
        self.project_id = project_id
        self.region = region
        self.model_display_name = model_display_name
        self.model: SVD = SVD()
        self.user_interactions_df: Optional[pd.DataFrame] = None
        self.local_model_filename = 'trained_svd_model.pkl'

    def load_data_from_gcs(self, gcs_data_path: str) -> bool: # Reads from GCS
        """Load interaction data from CSV file(s) in GCS."""
        print(f"Loading interaction data from GCS path: {gcs_data_path}")
        # Construct full path to the expected CSV file
        csv_file_path = f"{gcs_data_path.rstrip('/')}/interactions.csv"
        try:
            print(f"Attempting to read: {csv_file_path}")
            self.user_interactions_df = pd.read_csv(csv_file_path)

            if self.user_interactions_df.empty:
                print("Warning: Loaded DataFrame is empty.")
                return False

            required_cols = ['user_id', 'outfit_id', 'interaction']
            if not all(col in self.user_interactions_df.columns for col in required_cols):
                 raise ValueError(f"Input CSV must contain columns: {required_cols}")

            print(f"Loaded DataFrame with {len(self.user_interactions_df)} interactions from GCS.")
            return True

        except FileNotFoundError:
             print(f"ERROR: File not found at {csv_file_path}")
             return False
        except Exception as e:
            print(f"ERROR loading data from GCS: {e}")
            import traceback
            traceback.print_exc()
            return False

    def train_and_save_model_local(self) -> bool:
         # --- THIS FUNCTION REMAINS THE SAME (trains from the DataFrame) ---
         # (Code omitted for brevity - it's the same as in the Firestore version)
        if self.user_interactions_df is None or self.user_interactions_df.empty:
            print("No interaction data loaded, skipping model training.")
            return False
        try:
            print("Preparing data for Surprise...")
            self.user_interactions_df['user_id'] = self.user_interactions_df['user_id'].astype(str)
            self.user_interactions_df['outfit_id'] = self.user_interactions_df['outfit_id'].astype(str)
            self.user_interactions_df['interaction'] = pd.to_numeric(self.user_interactions_df['interaction'])
            min_rating = self.user_interactions_df['interaction'].min()
            max_rating = self.user_interactions_df['interaction'].max()
            print(f"Detected rating scale: min={min_rating}, max={max_rating}")
            if pd.isna(min_rating) or pd.isna(max_rating): raise ValueError("Cannot determine rating scale.")
            reader = Reader(rating_scale=(min_rating, max_rating))
            data = Dataset.load_from_df(self.user_interactions_df[['user_id', 'outfit_id', 'interaction']], reader)
            print("Building trainset...")
            trainset = data.build_full_trainset()
            print(f"Trainset built with {trainset.n_users} users and {trainset.n_items} items.")
            if trainset.n_users == 0 or trainset.n_items == 0:
                print("Warning: Trainset is empty. Skipping training.")
                return False
            print("Training SVD model...")
            self.model.fit(trainset)
            print(f"Saving trained model locally to {self.local_model_filename}...")
            with open(self.local_model_filename, 'wb') as f: pickle.dump(self.model, f)
            print("Model trained and saved locally successfully!")
            return True
        except Exception as e:
            print(f"ERROR during model training/local saving: {e}")
            import traceback; traceback.print_exc()
            return False

    def upload_model_to_gcs(self) -> Optional[str]:
        # --- THIS FUNCTION REMAINS THE SAME ---
        # (Code omitted for brevity)
        if not os.path.exists(self.local_model_filename): return None
        try:
            storage_client = storage.Client()
            if not self.model_output_dir.startswith("gs://"): raise ValueError("model_output_dir must be a GCS path.")
            bucket_name = self.model_output_dir.split('/')[2]
            blob_prefix = '/'.join(self.model_output_dir.split('/')[3:])
            if blob_prefix and not blob_prefix.endswith('/'): blob_prefix += '/'
            destination_blob_name = f"{blob_prefix}{self.local_model_filename}"
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(destination_blob_name)
            print(f"Uploading {self.local_model_filename} to {self.model_output_dir}{self.local_model_filename}...")
            blob.upload_from_filename(self.local_model_filename)
            print("Model uploaded to GCS successfully.")
            return self.model_output_dir
        except Exception as e:
            print(f"Error uploading model to GCS: {e}")
            return None

    def register_model_in_vertexai(self, model_gcs_uri: str):
         # --- THIS FUNCTION REMAINS THE SAME ---
         # (Code omitted for brevity)
        if not model_gcs_uri: return
        try:
            print(f"Registering model from {model_gcs_uri} in Vertex AI...")
            aiplatform.init(project=self.project_id, location=self.region)
            serving_image = 'us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest' # Example
            model = aiplatform.Model.upload(
                display_name=self.model_display_name,
                artifact_uri=model_gcs_uri,
                serving_container_image_uri=serving_image,
                description='SVD Collaborative Filtering model for Aura app (trained via custom job reading GCS)' # Updated description
            )
            print(f"Successfully registered model: {model.resource_name}")
        except Exception as e:
            print(f"ERROR registering model in Vertex AI: {e}")


# --- Main Execution Logic for Vertex AI Job ---
def main():
    parser = argparse.ArgumentParser(description='Train SVD recommender model from GCS CSV and register to Vertex AI')
    # ADDED BACK: --data-path argument
    parser.add_argument('--data-path', required=True, type=str, help='GCS path to the directory containing interactions.csv (e.g., gs://bucket/training-data/latest/)')
    parser.add_argument('--model-output-dir', required=True, type=str, help='GCS directory path to save the trained model artifact')
    parser.add_argument('--project-id', required=True, type=str, help='Google Cloud Project ID')
    parser.add_argument('--region', required=True, type=str, help='Google Cloud Region')
    parser.add_argument('--model-display-name', type=str, default='aura-svd-recommender', help='Display name for the model in Vertex AI Registry')

    args = parser.parse_args()
    print(f"Starting training job with args: {args}")

    updater = ModelUpdaterVertexGCS( # Using the GCS version
        model_output_dir=args.model_output_dir,
        project_id=args.project_id,
        region=args.region,
        model_display_name=args.model_display_name
    )

    # 1. Load data from GCS
    print("--- Step 1: Loading Data from GCS ---")
    if not updater.load_data_from_gcs(args.data_path):
        print("Failed to load data from GCS. Exiting.")
        exit(1) # Signal failure

    # 2. Train and save model locally
    print("\n--- Step 2: Training Model ---")
    if not updater.train_and_save_model_local():
        print("Failed to train/save model locally. Exiting.")
        exit(1)

    # 3. Upload model to GCS
    print("\n--- Step 3: Uploading Model to GCS ---")
    model_gcs_uri = updater.upload_model_to_gcs()
    if not model_gcs_uri:
        print("Failed to upload model to GCS. Exiting.")
        exit(1)

    # 4. Register model in Vertex AI
    print("\n--- Step 4: Registering Model in Vertex AI ---")
    updater.register_model_in_vertexai(model_gcs_uri)

    print("\nTraining job finished successfully.")

if __name__ == "__main__":
    main()
