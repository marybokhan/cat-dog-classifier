import pickle
import sys
import coremltools as ct
import torch
import os
from fastai.learner import Learner

# Define necessary functions for unpickling
def is_cat(x): 
    return True

class ModelConverter:
    def __init__(self, pickle_path="model.pkl", output_path="model.mlpackage"):
        self.pickle_path = pickle_path
        self.output_path = output_path
        
    def convert(self):
        """Convert pickle model to CoreML format"""
        print(f"Starting conversion: {self.pickle_path} → {self.output_path}")
        print(f"Model size: {os.path.getsize(self.pickle_path) / (1024*1024):.2f} MB")
        
        # Step 1: Load the pickle model into PyTorch format
        pytorch_model = self._load_pickle_to_torch()
        if pytorch_model is None:
            print("Failed to load model to PyTorch format")
            return False
            
        # Step 2: Convert PyTorch model to CoreML
        return self._convert_torch_to_coreml(pytorch_model)
    
    def _load_pickle_to_torch(self):
        """Load the pickle file into a PyTorch model"""
        print("Step 1: Loading pickle model to PyTorch...")
        
        # Register the is_cat function for unpickling
        sys.modules["__main__"].is_cat = is_cat
        
        try:
            # Method 1: Try direct loading with torch
            model = torch.load(self.pickle_path, map_location="cpu", weights_only=False)
            print(f"Model loaded successfully with torch.load: {type(model)}")
            
            # Extract the PyTorch model if it's a fastai Learner
            if isinstance(model, Learner):
                print("Extracting PyTorch model from fastai Learner")
                return model.model
            elif isinstance(model, torch.nn.Module):
                print("Model is already a PyTorch module")
                return model
            else:
                print(f"Unsupported model type: {type(model)}")
                return None
                
        except Exception as e1:
            print(f"Failed to load with torch.load: {e1}")
            
            try:
                # Method 2: Try with pickle unpickler
                class CustomUnpickler(pickle.Unpickler):
                    def find_class(self, module, name):
                        if module == "__main__" and name == "is_cat":
                            return is_cat
                        return super().find_class(module, name)
                
                with open(self.pickle_path, "rb") as f:
                    model = CustomUnpickler(f).load()
                
                print(f"Model loaded with custom unpickler: {type(model)}")
                
                # Extract the PyTorch model if it's a fastai Learner
                if hasattr(model, "model"):
                    print("Extracting PyTorch model from model object")
                    return model.model
                elif isinstance(model, torch.nn.Module):
                    return model
                else:
                    print(f"Unsupported model type: {type(model)}")
                    return None
                    
            except Exception as e2:
                print(f"Failed to load with custom unpickler: {e2}")
                return None
    
    def _convert_torch_to_coreml(self, pytorch_model):
        """Convert PyTorch model to CoreML format"""
        print("Step 2: Converting PyTorch model to CoreML...")
        
        try:
            # Put model in evaluation mode
            pytorch_model.eval()
            
            # Determine input shape - adjust as needed for your model
            input_shape = (1, 3, 224, 224)  # batch, channels, height, width
            print(f"Using input shape: {input_shape}")
            
            # Create example input for tracing
            example_input = torch.rand(*input_shape)
            
            # Trace the model for CoreML conversion
            print("Tracing the PyTorch model...")
            traced_model = torch.jit.trace(pytorch_model, example_input)
            
            # Convert to CoreML with image input
            print("Converting to CoreML format with image input...")
            mlmodel = ct.convert(
                traced_model,
                inputs=[ct.ImageType(name="input", shape=input_shape, bias=[-123.68, -116.779, -103.939], scale=1/255.0)]
            )
            
            # Save the CoreML model
            print(f"Saving CoreML model to {self.output_path}...")
            mlmodel.save(self.output_path)
            print("Conversion successful!")
            return True
            
        except Exception as e:
            print(f"Error during CoreML conversion: {e}")
            return False


if __name__ == "__main__":
    # Create and run the converter
    converter = ModelConverter("model.pkl", "model.mlpackage")
    success = converter.convert()
    
    if success:
        print("✅ Model conversion completed successfully")
    else:
        print("❌ Model conversion failed")
        sys.exit(1)
