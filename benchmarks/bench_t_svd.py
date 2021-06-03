"""Speed benchmark for Tucker HOOI decomposition.
"""
import tensornp as tnp
import sktensor as skt
import tensorly as tl
import tensorly.decomposition

from . import td_time_benchmark


def test_speed_3ord():
    """Decomposition on 3 order tensors.
    """
    print('\nSpeed benchmark for 3 order tensor svd.\n')

    print('----------TensorNP----------')
    td_time_benchmark(tnp.t_svd, tnp.tensor, max_iter=100, verbose=1)


def valid_3ord():
    print('\nCorrectness benchmark for 3 order tensor svd.\n')
    
    shape = (2, 3, 4)
    max_iter = 50
    print('----------TensorNP----------')
    norm_errors = 0
    for _ in range(max_iter):
        tensor = tnp.rand(2, 3, 4)
        u, s, v = tnp.t_svd(tensor)
        rec_tensor = tnp.reconstruct_t_svd(u, s, v)
        norm_error = tnp.linalg.norm(rec_tensor - tensor) / tnp.linalg.norm(tensor)
        norm_errors += norm_error
    print(f'error ({norm_errors/max_iter})')
