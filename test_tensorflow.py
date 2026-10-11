#!/usr/bin/env python3
"""
TensorFlow GPU Test Suite
Comprehensive tests for TensorFlow on NVIDIA L40S
"""

import time
import sys

def test_basic_operations():
    """Test basic TensorFlow GPU operations"""
    print("\n" + "="*70)
    print("  1. Basic Operations Test")
    print("="*70 + "\n")

    try:
        import tensorflow as tf

        with tf.device('/GPU:0'):
            # Tensor creation
            print("✓ Creating tensors on GPU...")
            x = tf.random.normal([1000, 1000])
            y = tf.random.normal([1000, 1000])

            # Basic operations
            print("✓ Testing addition...")
            z = tf.add(x, y)

            print("✓ Testing multiplication...")
            z = tf.multiply(x, y)

            print("✓ Testing matrix multiplication...")
            z = tf.matmul(x, y)

            print("✓ Testing reduction operations...")
            mean = tf.reduce_mean(z)
            std = tf.math.reduce_std(z)

            print(f"\n  Results:")
            print(f"    Mean: {mean.numpy():.4f}")
            print(f"    Std:  {std.numpy():.4f}")

            print("\n✓ All basic operations passed!\n")
            return True

    except Exception as e:
        print(f"\n✗ Basic operations failed: {e}\n")
        return False

def test_mixed_precision():
    """Test mixed precision capabilities"""
    print("\n" + "="*70)
    print("  2. Mixed Precision Test")
    print("="*70 + "\n")

    try:
        import tensorflow as tf

        # Test FP16
        print("Testing FP16 operations...")
        with tf.device('/GPU:0'):
            x_fp16 = tf.cast(tf.random.normal([2048, 2048]), tf.float16)
            y_fp16 = tf.cast(tf.random.normal([2048, 2048]), tf.float16)

            start = time.perf_counter()
            z_fp16 = tf.matmul(x_fp16, y_fp16)
            fp16_time = time.perf_counter() - start

            print(f"  FP16 matmul (2048x2048): {fp16_time*1000:.2f} ms")

        # Test FP32
        print("\nTesting FP32 operations...")
        with tf.device('/GPU:0'):
            x_fp32 = tf.random.normal([2048, 2048])
            y_fp32 = tf.random.normal([2048, 2048])

            start = time.perf_counter()
            z_fp32 = tf.matmul(x_fp32, y_fp32)
            fp32_time = time.perf_counter() - start

            print(f"  FP32 matmul (2048x2048): {fp32_time*1000:.2f} ms")

        print(f"\n  Speedup:")
        print(f"    FP16 vs FP32: {fp32_time/fp16_time:.2f}x faster")

        # Test mixed precision policy
        print("\nTesting mixed precision policy...")
        from tensorflow.keras import mixed_precision
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)

        print(f"  Compute dtype: {policy.compute_dtype}")
        print(f"  Variable dtype: {policy.variable_dtype}")

        # Reset policy
        mixed_precision.set_global_policy('float32')

        print("\n✓ Mixed precision tests passed!\n")
        return True

    except Exception as e:
        print(f"\n✗ Mixed precision test failed: {e}\n")
        return False

def test_neural_network():
    """Test neural network training"""
    print("\n" + "="*70)
    print("  3. Neural Network Training Test")
    print("="*70 + "\n")

    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

        # Build a simple CNN
        print("Creating model...")
        model = keras.Sequential([
            layers.Conv2D(64, 3, activation='relu', padding='same', input_shape=(224, 224, 3)),
            layers.MaxPooling2D(2),
            layers.Conv2D(128, 3, activation='relu', padding='same'),
            layers.MaxPooling2D(2),
            layers.Conv2D(256, 3, activation='relu', padding='same'),
            layers.MaxPooling2D(2),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Create dummy data
        batch_size = 32
        x_train = tf.random.normal([batch_size, 224, 224, 3])
        y_train = tf.random.uniform([batch_size], minval=0, maxval=10, dtype=tf.int32)

        print("Training for 10 iterations...")
        times = []

        for i in range(10):
            start = time.perf_counter()

            # Train step
            with tf.GradientTape() as tape:
                predictions = model(x_train, training=True)
                loss = tf.keras.losses.sparse_categorical_crossentropy(y_train, predictions)
                loss = tf.reduce_mean(loss)

            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            elapsed = time.perf_counter() - start
            times.append(elapsed)

            if i == 0 or (i + 1) % 5 == 0:
                print(f"  Iteration {i+1}: {elapsed*1000:.2f} ms, Loss: {loss.numpy():.4f}")

        avg_time = sum(times[1:]) / len(times[1:])
        print(f"\n  Average time (excl. warmup): {avg_time*1000:.2f} ms")
        print(f"  Throughput: {batch_size/avg_time:.2f} images/sec")

        print("\n✓ Neural network test passed!\n")
        return True

    except Exception as e:
        print(f"\n✗ Neural network test failed: {e}\n")
        return False

def test_dataset_pipeline():
    """Test tf.data pipeline on GPU"""
    print("\n" + "="*70)
    print("  4. Dataset Pipeline Test")
    print("="*70 + "\n")

    try:
        import tensorflow as tf

        # Create dataset
        print("Creating dataset pipeline...")
        dataset = tf.data.Dataset.from_tensor_slices(
            (tf.random.normal([1000, 224, 224, 3]),
             tf.random.uniform([1000], minval=0, maxval=10, dtype=tf.int32))
        )

        dataset = dataset.batch(32)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        # Simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

        # Test training
        print("Testing data pipeline with training...")
        start = time.perf_counter()

        history = model.fit(dataset, epochs=1, verbose=0)

        elapsed = time.perf_counter() - start

        print(f"  Training time: {elapsed:.2f} seconds")
        print(f"  Loss: {history.history['loss'][0]:.4f}")

        print("\n✓ Dataset pipeline test passed!\n")
        return True

    except Exception as e:
        print(f"\n✗ Dataset pipeline test failed: {e}\n")
        return False

def test_memory_info():
    """Test GPU memory information"""
    print("\n" + "="*70)
    print("  5. Memory Information Test")
    print("="*70 + "\n")

    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices('GPU')

        for gpu in gpus:
            print(f"GPU: {gpu.name}")

            # Get memory info
            details = tf.config.experimental.get_device_details(gpu)
            if 'device_name' in details:
                print(f"  Device: {details['device_name']}")

            # Memory growth
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"  Memory growth: Enabled")
            except:
                print(f"  Memory growth: Could not enable")

        # Test allocation
        print("\nTesting memory allocation...")
        with tf.device('/GPU:0'):
            tensors = []
            for i in range(5):
                t = tf.random.normal([1024, 1024, 256])
                tensors.append(t)
                print(f"  Allocated tensor {i+1}: {t.shape}")

        print("\n✓ Memory information test passed!\n")
        return True

    except Exception as e:
        print(f"\n✗ Memory information test failed: {e}\n")
        return False

def main():
    """Run all tests"""
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║       TensorFlow GPU Test Suite for NVIDIA L40S                 ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    try:
        import tensorflow as tf
    except ImportError:
        print("ERROR: TensorFlow is not installed!")
        print("Install with: pip install tensorflow")
        sys.exit(1)

    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("ERROR: No GPU devices found!")
        print("Please check your CUDA installation and GPU drivers.")
        sys.exit(1)

    print(f"\nTensorFlow Version: {tf.__version__}")
    print(f"GPUs Available: {len(gpus)}")
    for gpu in gpus:
        print(f"  - {gpu.name}")

    # Run tests
    tests = [
        ("Basic Operations", test_basic_operations),
        ("Mixed Precision", test_mixed_precision),
        ("Neural Network", test_neural_network),
        ("Dataset Pipeline", test_dataset_pipeline),
        ("Memory Information", test_memory_info),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"Unexpected error in {test_name}: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "="*70)
    print("  TEST SUMMARY")
    print("="*70 + "\n")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {test_name:25s} {status}")

    print(f"\n  Total: {passed}/{total} tests passed")
    print("="*70 + "\n")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
