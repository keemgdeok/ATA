import tensorflow as tf

# Check if TensorFlow is built with CUDA (GPU) support
print("Built with CUDA:", tf.test.is_built_with_cuda())

# Check if TensorFlow can access the GPU
print("GPU available:", tf.config.list_physical_devices('GPU'))

# Create a simple computation graph and run it on the GPU
with tf.device('/GPU:0'):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
    c = tf.matmul(a, b)

print("Result of matrix multiplication:\n", c)
