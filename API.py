from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional, Dict

# import pickle # REMOVED
import os
# import pandas as pd # REMOVED
# from surprise import Dataset, Reader, SVD # REMOVED

# --- Firebase Admin Setup ---
import firebase_admin
from firebase_admin import credentials, firestore, auth
from google.cloud import aiplatform # Added Vertex client

if not firebase_admin._apps:
    firebase_admin.initialize_app()
db = firestore.client()
# --- End Firebase Admin Setup ---

# --- Vertex AI Configuration ---
PROJECT_ID = os.environ.get('GCP_PROJECT')
REGION = os.environ.get('GCP_REGION', 'us-central1')
VERTEX_ENDPOINT_ID = os.environ.get('VERTEX_ENDPOINT_ID')
ENDPOINT_NAME = f"projects/{PROJECT_ID}/locations/{REGION}/endpoints/{VERTEX_ENDPOINT_ID}" if PROJECT_ID and REGION and VERTEX_ENDPOINT_ID else None
# --- End Vertex AI Configuration ---

app = FastAPI(title="AURA Recommender API - Cloud Run + Vertex AI") # Updated title

# --- Authentication Setup (No changes needed) ---
auth_scheme = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(auth_scheme)) -> str:
    token = credentials.credentials
    try:
        decoded_token = auth.verify_id_token(token)
        return decoded_token['uid']
    except auth.ExpiredIdTokenError:
         raise HTTPException(status_code=401, detail="Token expired", headers={"WWW-Authenticate": "Bearer"})
    except auth.InvalidIdTokenError as e:
         print(f"Token verification failed: {e}")
         raise HTTPException(status_code=401, detail="Invalid token", headers={"WWW-Authenticate": "Bearer"})
    except Exception as e:
        print(f"Authentication error: {e}")
        raise HTTPException(status_code=401, detail="Could not validate credentials", headers={"WWW-Authenticate": "Bearer"})
# --- End Authentication Setup ---

# --- Data Models (No changes needed) ---
class UserRequest(BaseModel):
    n_recommendations: Optional[int] = 10 # Keep this, but Vertex might return fixed number

class InteractionInput(BaseModel):
    outfit_id: str
    interaction: int

class InteractionRequest(BaseModel):
    interactions: List[InteractionInput]

class RecommendationResponse(BaseModel):
    recommendations: List[str] # Stays as list of outfit IDs
# --- End Data Models ---

# --- Global Variables (REMOVED model related) ---
# recommender: Optional[SVD] = None
# all_outfit_ids: set = set()
# --- End Global Variables ---


# --- Helper Function to Call Vertex AI ---
async def get_recommendations_from_vertex(user_id: str, n_recommendations: int) -> Optional[List[str]]:
    """Calls the Vertex AI endpoint to get outfit recommendations."""
    if not ENDPOINT_NAME:
         print("ERROR: Vertex AI Endpoint configuration missing (PROJECT_ID, REGION, VERTEX_ENDPOINT_ID env vars).")
         return None

    try:
        aiplatform.init(project=PROJECT_ID, location=REGION) # Initialize client here if needed

        # Prepare the prediction request payload
        # The format depends EXACTLY on the serving container signature.
        # For pre-built sklearn, it usually expects a list of instances.
        # Assuming the SVD model needs the user_id to predict.
        # NOTE: The deployed model might not directly accept n_recommendations.
        # It might return a ranked list, and we truncate it here.
        # Or, the custom serving container needs to handle it.
        # Let's assume it needs the user_id.
        instances = [{'user_id': user_id}] # Check your model serving signature!

        endpoint = aiplatform.Endpoint(endpoint_name=ENDPOINT_NAME)

        print(f"Calling Vertex AI Endpoint: {ENDPOINT_NAME} for user: {user_id}")
        prediction = endpoint.predict(instances=instances) # Send instance(s)

        # Process the prediction response
        # The structure prediction.predictions depends on your model output signature.
        # Example: Assuming it returns {'predictions': [{'recommended_outfit_ids': [id1, id2, ...]}]}
        if prediction.predictions and len(prediction.predictions) > 0:
             # Extract recommendations for the first (and likely only) instance
             # Adjust the key 'recommended_outfit_ids' based on actual model output!
             recommended_ids = prediction.predictions[0].get('recommended_outfit_ids', [])

             # Truncate to the requested number IF the model doesn't do it
             recommended_ids = recommended_ids[:n_recommendations]

             print(f"Received {len(recommended_ids)} recommendations from Vertex AI for user {user_id}")
             # Ensure IDs are strings
             return [str(id) for id in recommended_ids]
        else:
            print(f"No predictions received from Vertex AI for user {user_id}")
            return []

    except Exception as e:
        print(f"Error calling Vertex AI endpoint for user {user_id}: {e}")
        import traceback
        traceback.print_exc()
        return None # Indicate error

# --- API Events ---
@app.on_event("startup")
async def startup_event():
    # REMOVED model loading
    # REMOVED outfit ID loading
    print("API Startup: Initializing Firebase connection.")
    # Initialization happens globally now. Add checks if needed.
    if not ENDPOINT_NAME:
         print("WARNING: Vertex AI endpoint config is missing. /recommend endpoint will fail.")
    print("API Ready.")

@app.get("/")
async def root():
    # Simplified status check
    status = "healthy" if ENDPOINT_NAME else "degraded (missing Vertex AI config)"
    return {"status": status, "message": "AURA Recommender API (Cloud Run + Vertex AI)"}


@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(
    request: UserRequest, # Keep request body to get n_recommendations
    user_id: str = Depends(get_current_user)
):
    # REMOVED global recommender, all_outfit_ids checks

    if not ENDPOINT_NAME:
        raise HTTPException(status_code=503, detail="Recommendation service configuration missing.")

    print(f"Generating recommendations via Vertex AI for user: {user_id}")
    try:
        # --- Call Vertex AI Helper Function ---
        recommended_ids = await get_recommendations_from_vertex(user_id, request.n_recommendations)
        # --- End Vertex AI Call ---

        if recommended_ids is None: # Indicates an error occurred during Vertex call
            raise HTTPException(status_code=500, detail="Failed to get recommendations from prediction service.")

        if not recommended_ids:
            print(f"No recommendations found via Vertex AI for user {user_id}. Returning empty list.")
            # Consider fallback logic here if needed (e.g., popular items)

        # --- Return the list of IDs ---
        print(f"Returning {len(recommended_ids)} recommended outfit IDs.")
        return RecommendationResponse(recommendations=recommended_ids) # Sends List[str]

    except HTTPException: # Re-raise FastAPI/Auth exceptions
        raise
    except Exception as e: # Catch other unexpected errors
        print(f"ERROR in /recommend endpoint for user {user_id}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


# --- /update-interactions endpoint remains the same ---
@app.post("/update-interactions")
async def update_interactions(
    request: InteractionRequest,
    user_id: str = Depends(get_current_user)
):
    # ... (Keep the exact same implementation as your original api.py) ...
    print(f"Updating interactions for user: {user_id}")
    try:
        batch = db.batch()
        count = 0
        user_ref = db.collection('users').document(user_id)

        for interaction in request.interactions:
            # Check if outfit exists? Maybe not necessary here, training can filter later
            try:
                outfit_ref = db.collection('outfits').document(interaction.outfit_id)
                # Verify outfit exists (optional, adds read cost)
                # if not outfit_ref.get().exists:
                #     print(f"Warning: Outfit {interaction.outfit_id} not found. Skipping interaction.")
                #     continue
            except Exception as doc_ex:
                 print(f"Warning: Error creating reference for outfit {interaction.outfit_id}. Skipping. Error: {doc_ex}")
                 continue

            interaction_data = { "timestamp": firestore.SERVER_TIMESTAMP } # Use standard timestamp
            target_collection = None
            doc_id = f"{user_id}_{interaction.outfit_id}" # Consider composite ID to prevent duplicates per type?

            if interaction.interaction == 1: # Like
                target_collection = 'Likes'
                interaction_data['user_ref_like'] = user_ref
                interaction_data['outfit_ref_like'] = outfit_ref
                interaction_data['created_at_like'] = firestore.SERVER_TIMESTAMP # Use server timestamp
                # doc_ref = db.collection(target_collection).document(doc_id) # Example composite ID
                doc_ref = db.collection(target_collection).document() # Auto-ID (original)
            elif interaction.interaction == -1: # Dislike
                target_collection = 'Dislikes'
                interaction_data['user_ref_dislike'] = user_ref
                interaction_data['outfit_ref_dislike'] = outfit_ref
                interaction_data['created_at_dislike'] = firestore.SERVER_TIMESTAMP
                # doc_ref = db.collection(target_collection).document(doc_id)
                doc_ref = db.collection(target_collection).document()
            elif interaction.interaction == 3: # Wishlist
                target_collection = 'Wishlist'
                interaction_data['user_ref_wishlist'] = user_ref
                interaction_data['outfit_ref_wishlist'] = outfit_ref
                interaction_data['created_at_wishlist'] = firestore.SERVER_TIMESTAMP
                # doc_ref = db.collection(target_collection).document(doc_id)
                doc_ref = db.collection(target_collection).document()
            else:
                print(f"Warning: Skipping unknown interaction type {interaction.interaction} for outfit {interaction.outfit_id}")
                continue

            if target_collection:
                batch.set(doc_ref, interaction_data, merge=True) # Use set with merge=True if using composite IDs
                # batch.set(doc_ref, interaction_data) # Original
                count += 1

        if count > 0:
            batch.commit()
            print(f"Successfully processed {count} interactions for user {user_id}")
            return {"status": "success", "message": f"Processed {count} interactions."}
        else:
            print(f"No valid interactions provided for user {user_id}")
            return {"status": "no_op", "message": "No valid interactions were processed"}

    except Exception as e:
        print(f"ERROR updating interactions for user {user_id}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error while saving interactions: {e}")

# --- Main Execution (for testing locally - requires env vars) ---
if __name__ == "__main__":
    import uvicorn
    print("Starting Uvicorn server directly (for local testing)...")
    print("Ensure GCP_PROJECT, GCP_REGION, VERTEX_ENDPOINT_ID env vars are set.")
    print(f"Vertex Endpoint Config: {ENDPOINT_NAME}")
    # Port 8080 is default for Cloud Run
    uvicorn.run("api_cloudrun:app", host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), reload=True)