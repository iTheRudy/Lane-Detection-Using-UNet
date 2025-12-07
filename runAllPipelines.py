import train
import validate
import test

def main():

    print("\n============================")
    print("   LANE DETECTION PIPELINE  ")
    print("============================\n")

    print("1. Training model...\n")
    model_path, test_images_path, test_masks_path = train.run_training()

    print("\n2. Validating model...\n")
    val_iou = validate.run_validation(model_path, test_images_path, test_masks_path)
    print(f"Validation IoU: {val_iou:.4f}")

    print("\n3. Testing model...\n")
    test_results, files = test.run_testing(model_path, test_images_path, test_masks_path)

    print("\n====== TEST RESULTS ======")
    print(f"Test IoU:       {test_results['test_iou']:.4f}")
    print(f"Test Dice:      {test_results['test_dice']:.4f}")
    print(f"Test Accuracy:  {test_results['test_accuracy']:.4f}")

    print("\nSample prediction files:")
    for f in files:
        print(f" - {f}")

    print("\n============================")
    print(" Pipeline Execution Complete ")
    print("============================\n")


if __name__ == "__main__":
    main()
