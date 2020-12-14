import torch

def powerMethod(A, b, num_iter=1000, tol=1e-6, verbose=True):
	eig_old = torch.zeros(1)
	flag_tol_reached = False
	for it in range(num_iter):
		b = A(b)
		b = b / torch.norm(b)
		eig_max = torch.sum(b*A(b))
		if verbose:
			print('i:{0:3d} \t |e_new - e_old|:{1:2.2e}'.format(it,abs(eig_max-eig_old).item()))
		if abs(eig_max-eig_old)<tol:
			flag_tol_reached = True
			break
		eig_old = eig_max
	if verbose:
		print('tolerance reached!',it)
		print(f"L = {eig_max.item():.3e}")
	return eig_max.item(), b, flag_tol_reached
