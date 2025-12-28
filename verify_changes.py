from utils.model_handler import FoodModel
from utils.calorie_data import get_calorie_info
import sys

def verify():
    print("Testing Calorie Data...")
    foreign_items = ["pizza", "burger", "french_fries", "taco", "coke"]
    for item in foreign_items:
        cal = get_calorie_info(item)
        if cal:
            print(f"✅ Found calories for {item}: {cal}")
        else:
            print(f"❌ Failed to find calories for {item}")
            sys.exit(1)
            
    print("\nTesting Model Initialization...")
    try:
        model = FoodModel()
        if hasattr(model, 'base_model'):
            print("✅ Base Model (ImageNet) loaded successfully.")
        else:
            print("❌ Base Model failed to load.")
            sys.exit(1)
            
        # Check if custom model loaded (it might not if files are missing, but code shouldn't crash)
        if model.custom_model:
            print("ℹ️ Custom model also loaded.")
        else:
            print("ℹ️ Custom model NOT loaded (expected if files missing), but fallback is active.")
            
    except Exception as e:
        print(f"❌ Model initialization crashed: {e}")
        sys.exit(1)

    print("\n✅ Verification passed!")

if __name__ == "__main__":
    verify()
