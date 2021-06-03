import numpy as np


test_matrix = np.random.randn(100, 100)
# print(test_matrix)


u, s, v = np.linalg.svd(test_matrix)

print(u.shape, s.shape, v.shape)
for i in range(1, len(s) + 1):
    rec_matrix = u[:, :i] * s[:i] @ v[:i, :]
    err_matrix = test_matrix - rec_matrix
    print(f'stage [{i}]: relative error ({np.linalg.norm(err_matrix, ord=2)/np.linalg.norm(rec_matrix, ord=2):.3f}) '
          f'absolute error ({np.linalg.norm(err_matrix, ord=2):.3f})')
