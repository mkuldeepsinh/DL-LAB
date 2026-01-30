import tensorflow as tf

mat_a = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
mat_b = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)

add_tf = tf.add(mat_a, mat_b)
matmul_tf = tf.matmul(mat_a, mat_b)
transpose_tf = tf.transpose(mat_a)
det_tf = tf.linalg.det(mat_a)
inv_tf = tf.linalg.inv(mat_a)
eig_vals, eig_vecs = tf.linalg.eigh(mat_a)