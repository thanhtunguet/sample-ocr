from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

def load_image(image_path):
    return Image.open(image_path)

def extract_text_from_image(image_path, processor, model):
    # Load image
    image = load_image(image_path)
    
    # Preprocess the image
    pixel_values = processor(image, return_tensors="pt").pixel_values
    
    # Generate text
    generated_ids = model.generate(pixel_values)
    
    # Decode the generated text
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return generated_text

if __name__ == "__main__":
    # Path to the image
    image_path = 'test-03.jpg'  # Update with your image path

    # Load the processor and model
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
    
    # Extract text
    text = extract_text_from_image(image_path, processor, model)
    print("Extracted Text:\n", text)
