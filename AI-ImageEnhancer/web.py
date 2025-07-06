import streamlit as st
import cv2
import os
import tempfile
from realesrgan import RealESRGANer
import time
import random
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from streamlit_image_comparison import image_comparison

# Due to different versions of torchvision, it may cause errors: https://github.com/xinntao/Real-ESRGAN/issues/859
import sys
import types
try:
    from torchvision.transforms.functional_tensor import rgb_to_grayscale
except ImportError:
    # Fallback for older torchvision versions
    from torchvision.transforms.functional import rgb_to_grayscale
    functional_tensor = types.ModuleType("torchvision.transforms.functional_tensor")
    functional_tensor.rgb_to_grayscale = rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = functional_tensor


@st.cache_resource # Caches the model to avoid re-loading on every rerun
def load_model(model_name, device="cpu", tile=0):
    """
    Loads and initializes the RealESRGANer model.
    Caches the model to prevent repeated loading.
    """
    # Clear CUDA cache before loading a new model to free up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("CUDA cache cleared before model load.") # For debugging in console

    # Define model configurations: (model_architecture_instance, native_scale)
    model_configs = {
        'RealESRGAN_x4plus': (RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4), 4),
        'RealESRNet_x4plus': (RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4), 4),
        'RealESRGAN_x4plus_anime_6B': (RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4), 4),
        'RealESRGAN_x2plus': (RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2), 2)
    }

    if model_name not in model_configs:
        st.error(f'Unsupported model name: {model_name}. Please select a valid model.')
        raise ValueError(f'Unsupported model name {model_name}')

    model_arch_instance, netscale = model_configs[model_name]
    model_path = os.path.join('weights', model_name + '.pth')

    if not os.path.isfile(model_path):
        st.error(f'Model file `{model_path}` not found. Please ensure it is downloaded and placed in the `./weights/` directory.')
        raise FileNotFoundError(f'Model file {model_path} not found, please download it first')

    print(f'Attempting to load model {model_name} on device {device} with tile {tile}.')

    # Determine if half precision should be used (only for GPU)
    half = device != 'cpu' and torch.cuda.is_available()

    # Move model architecture to the specified device (CPU/GPU)
    model_arch_instance = model_arch_instance.to(device)
    # Apply half precision if running on GPU and enabled
    if half:
        model_arch_instance = model_arch_instance.half()
        print(f"Model architecture moved to {device} and set to half precision.")

    try:
        # Initialize RealESRGANer with the configured model architecture instance
        realesrgan_instance = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            model=model_arch_instance, # Pass the model architecture instance
            tile=tile,
            tile_pad=10,
            pre_pad=0,
            half=half,
            device=device
        )
        print(f"RealESRGANer instance created successfully for {model_name}.")
        return realesrgan_instance
    except Exception as e:
        st.error(f"Error initializing RealESRGANer: {e}. Check your PyTorch/CUDA setup and model files.")
        print(f"Error initializing RealESRGANer: {e}") # For debugging in console
        raise # Re-raise the exception to stop execution and show error in Streamlit


def main():
    """
    Main function to run the Streamlit application.
    """
    st.set_page_config(layout="wide") # Set page layout to wide

    # Create output folder if it doesn't exist
    output_folder = 'output'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Sidebar for controls
    st.sidebar.header("⚙️ Controls")

    # Mapping display names to actual model file names
    model_map = {
        'Upscaling x4 (General)': 'RealESRGAN_x4plus',
        'Upscaling x4 (Smoother)': 'RealESRNet_x4plus',
        'Upscaling x4 (Anime)': 'RealESRGAN_x4plus_anime_6B',
        'Upscaling x2 (General)': 'RealESRGAN_x2plus',
    }

    # Model selection dropdown
    display_name = st.sidebar.selectbox(
        "Select Model",
        list(model_map.keys())
    )
    model_name = model_map[display_name]

    # Determine available devices (CPU always, CUDA if available)
    available_devices = ['cpu']
    if torch.cuda.is_available():
        available_devices.insert(0, 'cuda:0') # Prioritize GPU if available

    # Device selection dropdown
    device_option = st.sidebar.selectbox(
        "Select Device",
        available_devices,
        help="Select 'cuda:0' if you have an NVIDIA GPU for faster processing."
    )
    # Tile parameter input
    tile = st.sidebar.number_input(
        "Tile Parameter",
        min_value=0,
        max_value=512,
        value=0,
        step=64,
        help="Split image to reduce GPU memory (0=No Split, often faster on CPU). Useful for large images."
    )

    # Initialize model_handler in session state if not present
    if 'model_handler' not in st.session_state:
        st.session_state.model_handler = None

    # Button to load the model
    if st.sidebar.button('💡 Load Model'):
        try:
            with st.spinner("Loading model... This might take a moment, especially for the first time."):
                # Call the cached load_model function
                st.session_state.model_handler = load_model(model_name, device=device_option, tile=tile)
                st.sidebar.success(f"Model '{display_name}' loaded successfully!")
        except Exception as e:
            # Display error in sidebar if model loading fails
            st.sidebar.error(f"Failed to load model: {e}")
            st.session_state.model_handler = None # Reset handler on failure

    # Sidebar instructions
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ How to Use:")
    st.sidebar.markdown("1. Select Model & Device.")
    st.sidebar.markdown("2. Click 'Load Model'.")
    st.sidebar.markdown("3. Upload an image below.")
    st.sidebar.markdown("4. Click 'Start Conversion'.")
    st.sidebar.markdown("5. View & Download!")

    # Main application title and description
    st.title("DetailForge ✨ AI Image Enhancer")
    st.markdown("Upload your low-resolution images and let AI enhance them!")
    st.divider()

    # File uploader for images
    uploaded_file = st.file_uploader("🖼️ Upload Your Image Here", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Create a unique temporary file path for the uploaded image
        file_extension = os.path.splitext(uploaded_file.name)[1]
        temp_input_filename = f"temp_input_{int(time.time())}_{random.randint(0, 1000)}{file_extension}"
        input_image_path = os.path.join(tempfile.gettempdir(), temp_input_filename)

        try:
            # Write the uploaded file content to the temporary path
            with open(input_image_path, "wb") as f:
                f.write(uploaded_file.read())
            print(f"Uploaded file saved to temporary path: {input_image_path}")

            # Read the image using OpenCV
            img = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)

            if img is None:
                st.error("Could not read the uploaded image. Please try another one or check its integrity.")
                print(f"Error: cv2.imread returned None for {input_image_path}")
            else:
                st.info("Image uploaded! Click 'Start Conversion' to enhance.")

                # Button to start conversion
                if st.button('🚀 Start Conversion'):
                    if st.session_state.model_handler is None:
                        st.error("🚨 Please load the model first using the button in the sidebar!")
                    else:
                        try:
                            with st.spinner('🪄 Working AI magic... Upscaling your image...'):
                                start_time = time.time()
                                # IMPORTANT FIX: Removed 'outscale' from enhance call.
                                # RealESRGANer is initialized with the correct scale.
                                # Passing 'outscale' here was likely causing the **kwargs error.
                                output, _ = st.session_state.model_handler.enhance(img)
                                end_time = time.time()

                                # Clear CUDA cache after enhancement to free up GPU memory
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                    print("CUDA cache cleared after enhancement.")

                                st.success(f"✅ Conversion Complete! (Took {end_time - start_time:.2f} seconds)")
                                st.markdown("### Compare Original vs. Upscaled (Use the slider!)")

                                # Display image comparison slider
                                image_comparison(
                                    img1=cv2.cvtColor(img, cv2.COLOR_BGR2RGB), # Convert BGR to RGB for correct display
                                    img2=cv2.cvtColor(output, cv2.COLOR_BGR2RGB), # Convert BGR to RGB for correct display
                                    label1="Original",
                                    label2="Upscaled"
                                )

                                st.divider()

                                # Ensure output directory exists before saving the output image
                                if not os.path.exists(output_folder):
                                    os.makedirs(output_folder)

                                # Generate a unique filename for the output image
                                filename = f"{int(time.time())}_{random.randint(0, 1000)}.png"
                                output_image_path = os.path.join(output_folder, filename)
                                cv2.imwrite(output_image_path, output) # Save the enhanced image
                                print(f"Output image saved to: {output_image_path}")

                                # Provide a download button for the enhanced image
                                with open(output_image_path, "rb") as file:
                                    st.download_button(
                                        label="📥 Download Upscaled Image (PNG)",
                                        data=file,
                                        file_name=filename,
                                        mime="image/png"
                                    )
                        except Exception as e:
                            # Catch and display errors during enhancement
                            st.error(f"An error occurred during image enhancement: {e}. Please try again or check your image/model.")
                            print(f"Error during enhancement: {e}") # For debugging in console
        finally:
            # Ensure temporary input file is removed after processing or error
            if os.path.exists(input_image_path):
                os.remove(input_image_path)
                print(f"Temporary input file removed: {input_image_path}")
    else:
        st.info("Waiting for you to upload an image...") # Message when no file is uploaded

    st.divider()
    # About This Project section
    with st.expander("📖 About This Project"):
        st.markdown("""
            This tool demonstrates the power of **Real-ESRGAN** (Enhanced Super-Resolution
            Generative Adversarial Networks) for upscaling images through a user-friendly
            web interface.

            Built with Python, Streamlit, and PyTorch, this mini-project aims to make cutting-edge
            AI image enhancement accessible. Simply upload an image, select a model from the sidebar,
            and see the magic happen!

            * **Original Real-ESRGAN Project:** [https://github.com/xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
            * **Built with Streamlit:** [https://streamlit.io/](https://streamlit.io/)
        """)

if __name__ == "__main__":
    main()

