import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def run_step(step_name, command):
    print("\n" + "=" * 60)
    print(f"🚀 {step_name}")
    print("=" * 60)

    result = subprocess.run(
        command,
        shell=True,
        cwd=PROJECT_ROOT
    )

    if result.returncode != 0:
        print(f"\n❌ Pipeline stopped at: {step_name}")
        sys.exit(1)

    print(f"\n✅ Finished: {step_name}")


if __name__ == "__main__":
    print("\n🛰️ RADAR CNN CLASSIFICATION PIPELINE STARTED")

    run_step("Step 1: Preprocess VOC into radar-like dataset", "python src/preprocess.py")
    run_step("Step 2: Train CNN model", "python src/train.py")
    run_step("Step 3: Predict test images", "python src/predict.py")

    print("\n🎉 FULL PIPELINE COMPLETED SUCCESSFULLY")