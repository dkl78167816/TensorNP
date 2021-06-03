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
    print('\nSpeed benchmark for 3 order tensor-train decomposition.\n')

    print('----------TensorNP----------')
    td_time_benchmark(tnp.tensor_ring, tnp.tensor, max_iter=100, verbose=1, 
                      ranks=[1, 2, 1])


def test_speed_8ord():
    """Decomposition on 8 order tensors.
    """
    print('\nSpeed benchmark for 8 order tensor-train decomposition.\n')

    print('----------TensorNP----------')
    td_time_benchmark(tnp.tensor_ring, tnp.tensor, max_iter=30, order=8, verbose=1, 
                      ranks=[1, 3, 4, 5, 6, 7, 8, 1])


def valid_3ord():
    print('\nCorrectness benchmark for 3 order tensor tucker decomposition.\n')
    
    shape = (2, 3, 4)
    max_iter = 100
    print('----------TensorNP----------')
    norm_errors = 0
    for _ in range(max_iter):
        tensor = tnp.randn(2, 3, 4)
        factors = tnp.tr_v2(tensor, ranks=[1, 2, 1])
        rec_tensor = tnp.reconstruct_tr(factors)
        norm_error = tnp.linalg.norm(rec_tensor - tensor) / tnp.linalg.norm(tensor)
        norm_errors += norm_error
    print(f'error ({norm_errors/max_iter})')
