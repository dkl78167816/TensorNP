# import sktensor as skt
# from tensornp.decomposition.cp import reconstruct_cp
# from tensornp.linalg import norm
# from tensornp.decomposition.hosvd import reconstruct_tucker 

# tensor = np.random.randn(2, 3, 4)
# sk_tensor = skt.dtensor(tensor)
# P, fit, itr = skt.cp_als(sk_tensor, 3, init='random')
# rec_tensor = reconstruct_cp(P.U, P.lmbda, (2, 3, 4))
# print(norm(tensor - rec_tensor)/norm(tensor))



# test_data1 = np.array([[[0, 1, 3, 4], [4, 0, 2, 1], [4, 2, 3, 4]],
#                         [[2, 4, 2, 3], [3, 3, 2, 4], [2, 3, 0, 2]]])
# sk_tensor = skt.dtensor(test_data1)
# factors, g = skt.tucker.hosvd(sk_tensor, [2, 3, 4], compute_core=True)
# res_data1 = reconstruct_tucker(g, factors)
# print(norm(res_data1 - test_data1)/norm(test_data1))

# sk_tensor = skt.dtensor(test_data1)
# g, factors = skt.tucker.hooi(sk_tensor, [2, 3, 4])
# res_data1 = reconstruct_tucker(g, factors)
# print(norm(res_data1 - test_data1)/norm(test_data1))

# import tensorly as tl
# import tensorly.decomposition

# g, factors = tensorly.decomposition.tucker(tl.tensor(test_data1), rank=[2, 3, 4])
# res_data1 = reconstruct_tucker(g, factors)
# print(norm(res_data1 - test_data1)/norm(test_data1))

# import numpy as np

# np.linalg.svd()

# from tensornp.linalg.tenalg import tensor_transpose
import numpy as np

# tensor = np.arange(1, 25).reshape(2, 3, 4)

t = np.ones((2, 3))

# print(tensor)
# print(tensor_transpose(tensor))


# np.pad()
# np.reshape()
