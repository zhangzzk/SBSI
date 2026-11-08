"""Four-state contrasts, with explicit target group and global normalization."""
import numpy as np

# Rows ++,+-,-+,--; columns target A,B; h=.02.
own=np.array([[1,1],[1,-1],[-1,1],[-1,-1]])/.08
opposite=np.array([[1,1],[-1,1],[1,-1],[-1,-1]])/.08

def contrasts(F):
    a=np.einsum('bsg,sg->b',F,own)
    b=np.einsum('bsg,sg->b',F,opposite)
    total=(F[:,0].sum(1)-F[:,3].sum(1))/.04
    np.testing.assert_allclose(a+b,total,rtol=0,atol=2e-12)
    return {'own_group':a,'opposite_group':b,'coherent_total':total}
