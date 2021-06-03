'''
@Author: duke
@Date: 2021-05-27 16:35:33
@LastEditTime: 2021-06-03 09:23:06
@LastEditors: duke
@Description: Benchmarks for tensor decomposition algorithms using TensorNP, scikit-tensor, Tensorly.
'''
import timeit
import tqdm
import numpy as np


def td_time_benchmark(f, initializer, order=3, max_iter=5, max_time=30, 
                 verbose=1, *args, **kw) -> float:
    """Do speed benchmark on given function. Return average time cost.

    Args:
        f (function): A python function to run with, receive tensor as parameter.
        initializer (function | class): Receive numpy array as raw data. Return specific type.
        max_iter (int, optional): Max iteration to do with function. Defaults to 5.
        max_time (int, optional): Max time to run with function. Defaults to 30.
        verbose (int, optional): How much information to log. Defaults to 0.
        *args, **kw: Parameters of f.
    """
    timer = timeit.default_timer
    used_time = 0

    if verbose > 1:
        pbar = tqdm.tqdm(total=max_iter)

    for i in range(max_iter):
        data_shape = list(range(2, order + 2))
        data = np.random.randn(*data_shape)
        tensor = initializer(data)
        
        tick = timer()
        f(tensor, *args, **kw)
        tock = timer()
        used_time += tock - tick

        if verbose > 1:
            pbar.update(1)

        if used_time > max_time:
            if verbose > 0:
                print(f'Iterations: [{i+1}] | Total: [{used_time:.5f}s] | Avg.: [{used_time/(i+1):.5f}s]')
            break
        
        if i == max_iter - 1:
            if verbose > 0:
                print(f'Iterations: [{i+1}] | Total: [{used_time:.5f}s] | Avg.: [{used_time/(i+1):.5f}s]')


# def td_valid_benchmark(f, initializer, reconstrut_f, order=3, max_iter=5, max_time=30, 
#                  verbose=1, *args, **kw) -> float:
#     """Do speed benchmark on given function. Return average time cost.

#     Args:
#         f (function): A python function to run with, receive tensor as parameter.
#         initializer (function | class): Receive numpy array as raw data. Return specific type.
#         reconstruct_f (function): Receive cores and factors to reconstruct tensor.
#         max_iter (int, optional): Max iteration to do with function. Defaults to 5.
#         max_time (int, optional): Max time to run with function. Defaults to 30.
#         verbose (int, optional): How much information to log. Defaults to 0.
#         *args, **kw: Parameters of f.
#     """
#     timer = timeit.default_timer
#     used_time = 0

#     if verbose > 1:
#         pbar = tqdm.tqdm(total=max_iter)

#     for i in range(max_iter):
#         data_shape = list(range(2, order + 2))
#         data = np.random.randn(*data_shape)
#         tensor = initializer(data)
        
#         tick = timer()
#         core, factors = f(tensor, *args, **kw)
#         rec_tensor = reconstrut_f(core, factors)
#         err = np.linalg.norm((rec_tensor - tensor).flatten(),)
#         tock = timer()
#         used_time += tock - tick

#         if verbose > 1:
#             pbar.update(1)

#         if used_time > max_time:
#             # if verbose > 0:
#             #     print(f'Iterations: [{i+1}] | Total: [{used_time:.5f}s] | Avg.: [{used_time/(i+1):.5f}s]')
#             break
        
#         if i == max_iter - 1:
#             # if verbose > 0:
#             #     print(f'Iterations: [{i+1}] | Total: [{used_time:.5f}s] | Avg.: [{used_time/(i+1):.5f}s]')