import subprocess
import sys


def main():
    print("=" * 70)
    print("🔥 OPTION FLOW SCANNER V3")
    print("=" * 70)

    print()
    print("[MAIN] STEP 1 MARKET REGIME START")
    print()

    result = subprocess.run(
        [
            sys.executable,
            "src/01_market_regime.py"
        ],
        check=False
    )

    if result.returncode != 0:
        print()
        print("[MAIN] STEP 1 FAILED")
        sys.exit(result.returncode)

    print()
    print("[MAIN] STEP 1 COMPLETE")
    print()


if __name__ == "__main__":
    main()
