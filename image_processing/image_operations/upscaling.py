import cv2
import os

# --- 1. Define Action Functions ---

def basic_upscale(img, output_path, size=(600, 600)):
    print("-> Executing Basic Interpolation (INTER_CUBIC)")
    upscaled_img = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(output_path, upscaled_img)
    print(f"-> Saved: {output_path}")

def basic_upscale_lanczos(img, output_path, size=(600, 600)):
    print("-> Executing Basic Interpolation (INTER_LANCZOS4)")
    upscaled_img = cv2.resize(img, size, interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(output_path, upscaled_img)
    print(f"-> Saved: {output_path}")

def dnn_upscale(img, output_path, model_name, model_file, scale):
    print(f"-> Executing DNN Super-Resolution with {model_name.upper()} (Scale x{scale})")

    if not os.path.exists(model_file):
        print(f"   ERROR: Model file not found at {model_file}. Skipping.")
        return

    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_file)
    sr.setModel(model_name, scale)

    upscaled_img_dl = sr.upsample(img)
    cv2.imwrite(output_path, upscaled_img_dl)
    print(f"-> Saved: {output_path}")

# --- 2. Define the Switch/Dispatch Dictionary (Actions Map) ---

def create_action_map(img):
    """Creates the dictionary mapping user choices to functions and arguments."""
    return {
        # Key: User Input Choice (e.g., "basic")
        "basic": (basic_upscale, img, "output_image/upscaled_output.jpg", (600, 600)),
        "lanczos": (basic_upscale_lanczos, img, "output_image/lanczos_upscaled_output.jpg", (600, 600)),
        "edsr": (dnn_upscale, img, "output_image/edsr_upscale_res.jpg", "edsr", "model/EDSR_x4.pb", 4),
        "lapsrn": (dnn_upscale, img, "output_image/lapsrn_upscale_res.jpg", "lapsrn", "model/LapSRN_x8.pb", 8),
    }

# --- 3. Main Logic Function ---

def run_upscale_from_user_input(img_path):
    """Loads image, gets user input, and executes the chosen upscaling method."""
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not load image at {img_path}. Please check the path and file.")
        return

    actions = create_action_map(img)
    
    # List available options for the user
    available_methods = list(actions.keys())
    
    # --- PROMPT USER INPUT ---
    print("\n--- Upscaling Method Selector ---")
    print("Available methods:")
    print(f"  {', '.join(available_methods)}")
    
    # Get user input
    user_choice = input("Enter the method you want to run: ").lower().strip()
    
    print("-" * 30)

    # The Dictionary-Based Switch-Case
    if user_choice in actions:
        # Get the function and arguments from the dictionary
        action_func, *args = actions[user_choice]
        
        # Execute the chosen function with its arguments
        action_func(*args)
    else:
        # Default/Else case
        print(f"Error: '{user_choice}' is not a valid method. Execution failed.")

# --- EXECUTION BLOCK ---

if __name__ == "__main__":
    # NOTE: Set the correct path to your test image here.
    IMAGE_PATH = "output_image/resized_image.png" 
    
    # This line now runs the function which contains the input() prompt.
    run_upscale_from_user_input(IMAGE_PATH)