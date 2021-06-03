"""Speed benchmark for CP Decomposition.
"""
import tensorly as tl
import tensorly.decomposition
import tensornp as tnp
import sktensor as skt

from . import td_time_benchmark


def test_speed_3ord():
    """Decomposition on 3 order tensors.
    """
    print('\nSpeed benchmark for 3 order tensor CP decomposition.\n')

    print('----------scikit-tensor----------')
    td_time_benchmark(skt.cp_als, skt.dtensor, max_iter=30, verbose=1, rank=3, init='random')
    print('----------TensorNP----------')
    td_time_benchmark(tnp.cp, tnp.tensor, max_iter=30, verbose=1, r=3, 
                 stop_iter=500, tol=1e-5, normalize_factor=True)
    print('----------Tensorly----------')
    td_time_benchmark(tensorly.decomposition.parafac, tl.tensor, max_iter=30, rank=3,
                 init='random', n_iter_max=500, tol=1e-5, normalize_factors=True)  

    # tl.decomposition.parafac  # 不可
    # import tensorly
    # tensorly.decomposition.parafac  # 不可

    # import tensorly.decomposition  
    # tensorly.decomposition.parafac  # 可

    # from tensorly import decomposition  # 可
    # decomposition.parafac


def test_speed_8ord():
    """Decomposition on 8 order tensors.
    """
    print('\nSpeed benchmark for 8 order tensor CP decomposition.\n')

    print('----------scikit-tensor----------')
    td_time_benchmark(skt.cp_als, skt.dtensor, max_iter=10, order=8, verbose=1, 
                 rank=8, init='random')
    print('----------TensorNP----------')
    td_time_benchmark(tnp.cp, tnp.tensor, max_iter=10, order=8, verbose=1, r=8, 
                 stop_iter=500, tol=1e-5, normalize_factor=True)
    print('----------Tensorly----------')
    td_time_benchmark(tensorly.decomposition.parafac, tl.tensor, max_iter=10, order=8, 
                 rank=8, init='random', n_iter_max=500, tol=1e-5, 
                 normalize_factors=True)  


def valid_3ord():
    print('\nCorrectness benchmark for 3 order tensor CP decomposition.\n')
    
    shape = (2, 3, 4)
    max_iter = 30
    print('----------TensorNP----------')
    norm_errors = 0
    for _ in range(max_iter):
        tensor = tnp.randn(2, 3, 4)
        factors, lamda = tnp.cp(tensor, r=3, stop_iter=500, tol=1e-5, normalize_factor=True)
        rec_tensor = tnp.reconstruct_cp(factors, lamda, shape)
        norm_error = tnp.linalg.norm(rec_tensor - tensor) / tnp.linalg.norm(tensor)
        norm_errors += norm_error
    print(f'error ({norm_errors/max_iter})')

    print('----------scikit-tensor----------')
    norm_errors = 0
    for _ in range(max_iter):
        tensor = tnp.randn(2, 3, 4)
        skt_tensor = skt.dtensor(tensor)
        P, _, _ = skt.cp_als(skt_tensor, rank=3, init='random')
        rec_tensor = P.toarray()
        norm_error = tnp.linalg.norm(rec_tensor - skt_tensor) / tnp.linalg.norm(tensor)
        norm_errors += norm_error
    print(f'error ({norm_errors/max_iter})')


    print('----------Tensorly----------')
    norm_errors = 0
    for _ in range(max_iter):
        tensor = tnp.randn(2, 3, 4)
        tl_tensor = tl.tensor(tensor)
        cp_tensor = tensorly.decomposition.parafac(
            tl_tensor, rank=3, n_iter_max=500, tol=1e-6, normalize_factors=True, init='random')
        rec_tensor = tnp.reconstruct_cp(cp_tensor.factors, cp_tensor.weights, shape)
        norm_error = tnp.linalg.norm(rec_tensor - tensor) / tnp.linalg.norm(tensor)
        norm_errors += norm_error
    print(f'error ({norm_errors/max_iter})')
