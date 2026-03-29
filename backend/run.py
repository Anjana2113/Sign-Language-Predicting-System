"""
run.py — Main Entry Point for SignAI Project
"""
import os
import sys

def check_environment():
    """Verify that all required files and models exist before starting."""
    print("\n" + "="*50)
    print(" Initializing SignAI Assistive Interpreter...")
    print("="*50)
    
    missing_files = []
    
    # Check for .env
    if not os.path.exists(".env"):
        print(" [WARN] No .env file found. Copying from .env.example...")
        try:
            with open(".env.example", "r") as f:
                with open(".env", "w") as out:
                    out.write(f.read())
            print(" [OK] Created .env file.")
        except:
            pass

    # Import config to check paths
    import config
    
    required_models = [
        config.LANDMARK_MODEL_PATH,
        config.LABEL_MAP_PATH,
        os.path.join(config.MODELS_DIR, "hand_landmarker.task"),
        os.path.join(config.BASE_DIR, "animation_landmarks_data.npy")
    ]
    
    for path in required_models:
        if not os.path.exists(path):
            missing_files.append(os.path.basename(path))
            
    if missing_files:
        print("\n [ERROR] Missing critical model files:")
        for mf in missing_files:
            print(f"   - {mf}")
        print("\nPlease ensure you have downloaded all models to the correct directories.")
        sys.exit(1)
        
    print(" [OK] All models and configurations found.\n")

if __name__ == "__main__":
    check_environment()
    
    # Start the Flask app
    from src.web_server import app
    print("\n   Server starting at: http://localhost:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
