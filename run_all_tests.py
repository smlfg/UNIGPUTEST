#!/usr/bin/env python3
"""
Complete LLM Testing Suite - Master Script
===========================================

Führt alle Tests nacheinander aus:
1. Advanced Models Performance Test
2. Prompt Quality Evaluation

Estimated total time: 30-45 minutes
"""

import subprocess
import sys
import time
from datetime import datetime

def print_header(title: str, char: str = "="):
    """Schöner Header"""
    print(f"\n{char*80}")
    print(f"  {title}")
    print(f"{char*80}\n")

def run_script(script_name: str, description: str):
    """Führt ein Python-Script aus"""
    print_header(f"RUNNING: {description}")

    print(f"Script: {script_name}")
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}\n")

    start = time.time()

    try:
        # Run script
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            text=True
        )

        elapsed = time.time() - start

        print(f"\n✅ {description} completed!")
        print(f"⏱️  Time: {elapsed/60:.1f} minutes")

        return True

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start

        print(f"\n❌ {description} failed!")
        print(f"⏱️  Time: {elapsed/60:.1f} minutes")
        print(f"Error: {e}")

        return False

def main():
    """Hauptprogramm"""
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║            Complete LLM Testing Suite - Master Runner               ║
    ║            NVIDIA L40S - Full Model Evaluation                      ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)

    overall_start = time.time()

    print(f"🚀 Starting complete test suite at {datetime.now().strftime('%H:%M:%S')}")
    print(f"📊 Estimated total time: 30-45 minutes\n")

    # Test Suite
    tests = [
        ("test_advanced_models.py", "Advanced Models Performance Test"),
        ("test_prompt_quality.py", "Prompt Quality Evaluation"),
    ]

    results = {}

    # Run all tests
    for script, description in tests:
        success = run_script(script, description)
        results[description] = success

        if success:
            print(f"\n⏸️  Pausing 10 seconds before next test...\n")
            time.sleep(10)
        else:
            print(f"\n⚠️  Continuing to next test despite failure...\n")
            time.sleep(5)

    # Final Summary
    overall_time = time.time() - overall_start

    print_header("COMPLETE TEST SUITE SUMMARY")

    print(f"Total Time: {overall_time/60:.1f} minutes\n")

    print("Test Results:")
    print("─" * 80)

    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {test_name:<50} {status}")

    print("─" * 80)

    passed = sum(1 for s in results.values() if s)
    total = len(results)

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests completed successfully!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check output above for details.")

    print("\n" + "="*80)
    print(f"Finished at {datetime.now().strftime('%H:%M:%S')}")
    print("="*80 + "\n")

    print("""
    📁 Generated Files:
       - llm_benchmark_results.json (if ran earlier)
       - prompt_quality_results.json (from quality test)

    🔍 Next Steps:
       1. Review the prompt quality JSON for detailed comparisons
       2. Try visualizations: python visualize_results.py
       3. Test with your own prompts!
    """)

if __name__ == "__main__":
    main()
