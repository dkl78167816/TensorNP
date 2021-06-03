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
    print('\nSpeed benchmark for 3 order tensor Tucker decomposition.\n')

    print('----------TensorNP----------')
    td_time_benchmark(tnp.tucker, tnp.tensor, max_iter=30, verbose=1, 
                      ranks=[2, 3, 4], tol=1e-6, stop_iter=500)
    print('----------scikit-tensor----------')
    td_time_benchmark(skt.tucker.hooi, skt.dtensor, max_iter=30, verbose=1, 
                 rank=[2, 3, 4], init='nvecs')
    print('----------Tensorly----------')
    td_time_benchmark(tensorly.decomposition.tucker, tl.tensor, max_iter=30, 
                 verbose=1, rank=[2, 3, 4], n_iter_max=500, tol=1e-6)


def test_speed_8ord():
    """Decomposition on 8 order tensors.
    """
    print('\nSpeed benchmark for 8 order tensor Tucker decomposition.\n')

    print('----------TensorNP----------')
    td_time_benchmark(tnp.tucker, tnp.tensor, max_iter=10, order=8, verbose=1, 
                      ranks=list(range(2, 10)), tol=1e-6, stop_iter=500)
    print('----------scikit-tensor----------')
    td_time_benchmark(skt.tucker.hooi, skt.dtensor, max_iter=10, order=8, verbose=1, 
                 rank=list(range(2, 10)), init='nvecs')
    print('----------Tensorly----------')
    td_time_benchmark(tensorly.decomposition.tucker, tl.tensor, max_iter=10, order=8, 
                 verbose=1, rank=list(range(2, 10)), n_iter_max=500, tol=1e-6)


def valid_3ord():
    print('\nCorrectness benchmark for 3 order tensor tucker decomposition.\n')
    
    shape = (2, 3, 4)
    max_iter = 30
    print('----------TensorNP----------')
    norm_errors = 0
    for _ in range(max_iter):
        tensor = tnp.randn(2, 3, 4)
        g, factors = tnp.tucker(tensor, ranks=[2, 3, 4], tol=1e-6, stop_iter=500)
        rec_tensor = tnp.reconstruct_hosvd(g, factors)
        norm_error = tnp.linalg.norm(rec_tensor - tensor) / tnp.linalg.norm(tensor)
        norm_errors += norm_error
    print(f'error ({norm_errors/max_iter})')

    print('----------scikit-tensor----------')
    norm_errors = 0
    for _ in range(max_iter):
        tensor = tnp.randn(2, 3, 4)
        skt_tensor = skt.dtensor(tensor)
        g, factors = skt.tucker.hooi(skt_tensor, rank=[2, 3, 4], init='nvecs')
        rec_tensor = tnp.reconstruct_hosvd(g, factors)
        norm_error = tnp.linalg.norm(rec_tensor - tensor) / tnp.linalg.norm(tensor)
        norm_errors += norm_error
    print(f'error ({norm_errors/max_iter})')


    print('----------tensorly----------')
    norm_errors = 0
    for _ in range(max_iter):
        tensor = tnp.randn(2, 3, 4)
        tl_tensor = tl.tensor(tensor)
        tucker_tensor = tensorly.decomposition.tucker(tl_tensor, rank=[2, 3, 4], n_iter_max=500, tol=1e-6)
        rec_tensor = tnp.reconstruct_hosvd(tucker_tensor.core, tucker_tensor.factors)
        norm_error = tnp.linalg.norm(rec_tensor - tensor) / tnp.linalg.norm(tensor)
        norm_errors += norm_error
    print(f'error ({norm_errors/max_iter})')
