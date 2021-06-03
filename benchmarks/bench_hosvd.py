"""Speed benchmark for HOSVD.
"""
import tensornp as tnp
import sktensor as skt

from . import td_time_benchmark


def test_speed_3ord():
    """Decomposition on 3 order tensors.
    """
    print('\nSpeed benchmark for 3 order tensor HOSVD.\n')

    print('----------TensorNP----------')
    td_time_benchmark(tnp.hosvd, tnp.tensor, max_iter=30, verbose=1)
    print('----------scikit-tensor----------')
    td_time_benchmark(skt.tucker.hosvd, skt.dtensor, max_iter=30, verbose=1, 
                 rank=[2, 3, 4])


def test_speed_8ord():
    """Decomposition on 8 order tensors.
    """
    print('\nSpeed benchmark for 8 order tensor HOSVD.\n')

    print('----------TensorNP----------')
    td_time_benchmark(tnp.hosvd, tnp.tensor, order=8, max_iter=30, verbose=1)
    print('----------scikit-tensor----------')
    td_time_benchmark(skt.tucker.hosvd, skt.dtensor, order=8, max_iter=30, verbose=1, 
                 rank=list(range(2, 10)))


def valid_3ord():
    print('\nCorrectness benchmark for 3 order tensor HOSVD.\n')
    
    shape = (2, 3, 4)
    max_iter = 30
    print('----------TensorNP----------')
    norm_errors = 0
    for _ in range(max_iter):
        tensor = tnp.randn(2, 3, 4)
        g, factors = tnp.hosvd(tensor, compute_core=True)
        rec_tensor = tnp.reconstruct_hosvd(g, factors)
        norm_error = tnp.linalg.norm(rec_tensor - tensor) / tnp.linalg.norm(tensor)
        norm_errors += norm_error
    print(f'error ({norm_errors/max_iter})')

    print('----------scikit-tensor----------')
    norm_errors = 0
    for _ in range(max_iter):
        tensor = tnp.randn(2, 3, 4)
        skt_tensor = skt.dtensor(tensor)
        factors, g = skt.tucker.hosvd(skt_tensor, rank=[2, 3, 4])
        rec_tensor = tnp.reconstruct_hosvd(g, factors)
        norm_error = tnp.linalg.norm(rec_tensor - tensor) / tnp.linalg.norm(tensor)
        norm_errors += norm_error
    print(f'error ({norm_errors/max_iter})')
