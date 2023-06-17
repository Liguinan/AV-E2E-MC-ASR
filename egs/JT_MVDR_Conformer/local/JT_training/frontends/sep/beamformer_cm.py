import torch
from torch_complex import functional as FC
from torch_complex.tensor import ComplexTensor
#import pudb
# from params import * 

def  get_power_spectral_density_matrix_self_with_cm(xs: ComplexTensor, mask: ComplexTensor,
                                      averaging=True,
                                      normalization=True,
                                      eps: float = 1e-6
                                      ) -> ComplexTensor:
    """Return cross-channel power spectral density (PSD) matrix
    Args:
        xs (ComplexTensor): (..., F, C, T)
        mask (torch.Tensor): (..., F, C, T)
        normalization (bool):
        eps (float):
    Returns
        psd (ComplexTensor): (..., F, C, C)
    """
    # outer product: (..., C_1, T) x (..., C_2, T) -> (..., T, C, C_2)
    psd = FC.einsum('...ct,...et->...tce', [xs, xs.conj()])

    # Averaging mask along C: (..., C, T) -> (..., T)
    #mask = mask.mean(dim=-2) # by yong, no multi channel mask for the moment @2019.11.22

    # Normalized mask along T: (..., T)
    #if normalization:
    #    # If assuming the tensor is padded with zero, the summation along
    #    # the time axis is same regardless of the padding length.
    #    mask = mask / (mask.sum(dim=-1, keepdim=True) + eps)

    ## psd: (..., T, C, C)
    #psd = psd_Y * mask[..., None, None]
   
    #normalize psd with mask:
    #mask=mask*mask
    mask=mask.real*mask.real+mask.imag*mask.imag
    norm_mask=(mask.sum(dim=-1, keepdim=True) + eps)
    psd = psd / norm_mask[..., None, None]

    # (..., T, C, C) -> (..., C, C)
    psd = psd.sum(dim=-3)
    #psd = psd.mean(dim=-3)

    return psd 





def  get_power_spectral_density_matrix_self(xs: ComplexTensor) -> ComplexTensor:
    """Return cross-channel power spectral density (PSD) matrix
    Args:
        xs (ComplexTensor): (..., F, C, T)
        mask (torch.Tensor): (..., F, C, T)
        normalization (bool):
        eps (float):
    Returns
        psd (ComplexTensor): (..., F, C, C)
    """
    # outer product: (..., C_1, T) x (..., C_2, T) -> (..., T, C, C_2)
    psd = FC.einsum('...ct,...et->...tce', [xs, xs.conj()])

    # Averaging mask along C: (..., C, T) -> (..., T)
    #mask = mask.mean(dim=-2) # by yong, no multi channel mask for the moment @2019.11.22

    # Normalized mask along T: (..., T)
    #if normalization:
    #    # If assuming the tensor is padded with zero, the summation along
    #    # the time axis is same regardless of the padding length.
    #    mask = mask / (mask.sum(dim=-1, keepdim=True) + eps)

    ## psd: (..., T, C, C)
    #psd = psd_Y * mask[..., None, None]
    # (..., T, C, C) -> (..., C, C)
    psd = psd.sum(dim=-3)

    return psd


def get_power_spectral_density_matrix(xs: ComplexTensor, mask: torch.Tensor,
                                      averaging=True,
                                      normalization=True,
                                      eps: float = 1e-6
                                      ) -> ComplexTensor:
    """Return cross-channel power spectral density (PSD) matrix

    Args:
        xs (ComplexTensor): (..., F, C, T)
        mask (torch.Tensor): (..., F, C, T)
        normalization (bool):
        eps (float):
    Returns
        psd (ComplexTensor): (..., F, C, C)

    """
    # outer product: (..., C_1, T) x (..., C_2, T) -> (..., T, C, C_2)
    psd_Y = FC.einsum('...ct,...et->...tce', [xs, xs.conj()])

    # Averaging mask along C: (..., C, T) -> (..., T)
    #if averaging:
    #    mask = mask.mean(dim=-2) #commented out by yongxu, @2019.11.26, as there is only one chnanel mask for the moment

    # Normalized mask along T: (..., T)
    if normalization:
        # If assuming the tensor is padded with zero, the summation along
        # the time axis is same regardless of the padding length.
        mask = mask / (mask.sum(dim=-1, keepdim=True) + eps)
        #mask = mask + eps * torch.rand_like(mask)

    # psd: (..., T, C, C)
    psd = psd_Y * mask[..., None, None]
    # (..., T, C, C) -> (..., C, C)
    psd = psd.sum(dim=-3)

    return psd


def get_mvdr_vector(psd_s: ComplexTensor,
                    psd_n: ComplexTensor,
                    reference_vector: torch.Tensor,
                    eps: float = 10e-6
                    ) -> ComplexTensor:
    """Return the MVDR(Minimum Variance Distortionless Response) vector:

        h = (Npsd^-1 @ Spsd) / (Tr(Npsd^-1 @ Spsd)) @ u

    Reference:
        On optimal frequency-domain multichannel linear filtering
        for noise reduction; M. Souden et al., 2010;
        https://ieeexplore.ieee.org/document/5089420

    Args:
        psd_s (ComplexTensor): (..., F, C, C)
        psd_n (ComplexTensor): (..., F, C, C)
        reference_vector (torch.Tensor): (..., C)
        eps (float):
    Returns:
        beamform_vector (ComplexTensor)r: (..., F, C)
    """
    # Add eps
    C = psd_n.size(-1)
    eye = torch.eye(C, dtype=psd_n.dtype, device=psd_n.device)
    shape = [1 for _ in range(psd_n.dim() - 2)] + [C, C]
    eye = eye.view(*shape)
    psd_n += eps * eye
    #import pdb; pdb.set_trace()
    try:
        psd_n_i = psd_n.inverse()
    except:
        try:
            reg_coeff_tensor = ComplexTensor(torch.rand_like(psd_n.real),
                                             torch.rand_like(psd_n.real))*1e-2
            psd_n = psd_n/10e+4
            psd_s = psd_s/10e+4
            psd_n += reg_coeff_tensor
            psd_n_i = psd_n.inverse()
        except:
            try:
                reg_coeff_tensor = ComplexTensor(torch.rand_like(psd_n.real),
                                                 torch.rand_like(psd_n.real))*1e-1
                psd_n = psd_n/10e+10
                psd_s = psd_s/10e+10
                psd_n += reg_coeff_tensor
                psd_n_i = psd_n.inverse()
            except:
                try:
                    reg_coeff_tensor = ComplexTensor(torch.rand_like(psd_n.real),
                                                     torch.rand_like(psd_n.real))*1e-1
                    psd_n = psd_n/10e+10
                    psd_s = psd_s/10e+10
                    psd_n += reg_coeff_tensor
                    psd_n_i = psd_n.inverse()
                except:
                    reg_coeff_tensor = ComplexTensor(torch.rand_like(psd_n.real),
                                                     torch.rand_like(psd_n.real))
                    psd_n = psd_n/10e+10
                    psd_s = psd_s/10e+10
                    psd_n += reg_coeff_tensor
                    psd_n_i = psd_n.inverse()


    # numerator: (..., C_1, C_2) x (..., C_2, C_3) -> (..., C_1, C_3)
    numerator = FC.einsum('...ec,...cd->...ed', [psd_n_i, psd_s])
    # ws: (..., C, C) / (...,) -> (..., C, C)
    ws = numerator / (FC.trace(numerator)[..., None, None] + eps)
    # h: (..., F, C_1, C_2) x (..., C_2) -> (..., F, C_1)
    beamform_vector = FC.einsum('...fec,...c->...fe', [ws, reference_vector])
    return beamform_vector


def get_mvdr_vector_stable(
    psd_s: ComplexTensor,
    psd_n: ComplexTensor,
    reference_vector: torch.Tensor,
    use_diag_loading: bool = True,
    diag_loading_ratio: float = 1e-5,
    use_torch_solver: bool = True,
    eps = 1e-5
) -> ComplexTensor:
    """Return the MVDR(Minimum Variance Distortionless Response) vector:

        h = (Npsd^-1 @ Spsd) / (Tr(Npsd^-1 @ Spsd)) @ u

    Reference:
        On optimal frequency-domain multichannel linear filtering
        for noise reduction; M. Souden et al., 2010;
        https://ieeexplore.ieee.org/document/5089420

    Args:
        psd_s (ComplexTensor): (..., F, C, C)
        psd_n (ComplexTensor): (..., F, C, C)
        reference_vector (torch.Tensor): (..., C)
        eps (float):
    Returns:
        beamform_vector (ComplexTensor)r: (..., F, C)
    """
    # Add eps
    B, F = psd_n.shape[:2]
    C = psd_n.size(-1)
    eye = torch.eye(C, dtype=psd_n.dtype, device=psd_n.device)
    shape = [1 for _ in range(psd_n.dim() - 2)] + [C, C]
    eye = eye.view(*shape).repeat(B, F, 1, 1)
    epsilon = None
    if use_diag_loading:
        with torch.no_grad():
            epsilon = FC.trace(psd_n).real.abs()[..., None, None] * diag_loading_ratio
            # in case that correlation_matrix is all-zero
            # import pdb; pdb.set_trace()
            epsilon += diag_loading_ratio
    else: 
        epsilon = diag_loading_ratio

    if use_torch_solver:
        # import pdb; pdb.set_trace()
        print("use_torch_solver")
        numerator = FC.solve(psd_s, psd_n + epsilon * eye)
    else:
        psd_n += epsilon * eye
        try:
            psd_n_i = psd_n.inverse()
        except:
            try:
                reg_coeff_tensor = ComplexTensor(torch.rand_like(psd_n.real), torch.rand_like(psd_n.real)) * 1e-2
                psd_n = psd_n / 10e+4
                psd_s = psd_s / 10e+4
                psd_n += reg_coeff_tensor
                psd_n_i = psd_n.inverse()
            except:
                try:
                    reg_coeff_tensor = ComplexTensor(torch.rand_like(psd_n.real), torch.rand_like(psd_n.real)) * 1e-1
                    psd_n = psd_n / 10e+10
                    psd_s = psd_s / 10e+10
                    psd_n += reg_coeff_tensor
                    psd_n_i = psd_n.inverse()
                except:
                    try:
                        reg_coeff_tensor = ComplexTensor(torch.rand_like(psd_n.real),
                                                         torch.rand_like(psd_n.real)) * 1e-1
                        psd_n = psd_n / 10e+10
                        psd_s = psd_s / 10e+10
                        psd_n += reg_coeff_tensor
                        psd_n_i = psd_n.inverse()
                    except:
                        reg_coeff_tensor = ComplexTensor(torch.rand_like(psd_n.real), torch.rand_like(psd_n.real))
                        psd_n = psd_n / 10e+10
                        psd_s = psd_s / 10e+10
                        psd_n += reg_coeff_tensor
                        psd_n_i = psd_n.inverse()

        # numerator: (..., C_1, C_2) x (..., C_2, C_3) -> (..., C_1, C_3)
        # [32, 257, 15, 15]
        numerator = FC.einsum('...ec,...cd->...ed', [psd_n_i, psd_s])

    # ws: (..., C, C) / (...,) -> (..., C, C)
    ws = numerator / (FC.trace(numerator)[..., None, None] + eps)
    # h: (..., F, C_1, C_2) x (..., C_2) -> (..., F, C_1)
    beamform_vector = FC.einsum("...fec,...c->...fe", [ws, reference_vector])
    return beamform_vector




def get_smvdr_vector(psd_s: ComplexTensor,
                     psd_n: ComplexTensor,
                     psd_i: ComplexTensor,
                     reference_vector: torch.Tensor,
                     eps: float = 1e-15) -> ComplexTensor:
    C = psd_n.size(-1)
    eye = torch.eye(C, dtype=psd_n.dtype, device=psd_n.device)
    shape = [1 for _ in range(psd_n.dim() - 2)] + [C, C]
    eye = eye.view(*shape)
    psd_n += eps * eye
    psd_i += eps * eye
    psd_n += psd_i
    try:
        psd_n_i = psd_n.inverse()
    except:
        try:
            reg_coeff_tensor = ComplexTensor(torch.rand_like(psd_n.real),
                                             torch.rand_like(psd_n.real))*1e-2
            psd_n = psd_n/10e+4
            psd_s = psd_s/10e+4
            psd_n += reg_coeff_tensor
            psd_n_i = psd_n.inverse()
        except:
            try:
                reg_coeff_tensor = ComplexTensor(torch.rand_like(psd_n.real),
                                                 torch.rand_like(psd_n.real))*1e-1
                psd_n = psd_n/10e+10
                psd_s = psd_s/10e+10
                psd_n += reg_coeff_tensor
                psd_n_i = psd_n.inverse()
            except:
                try:
                    reg_coeff_tensor = ComplexTensor(torch.rand_like(psd_n.real),
                                                     torch.rand_like(psd_n.real))*1e-1
                    psd_n = psd_n/10e+10
                    psd_s = psd_s/10e+10
                    psd_n += reg_coeff_tensor
                    psd_n_i = psd_n.inverse()
                except:
                    reg_coeff_tensor = ComplexTensor(torch.rand_like(psd_n.real),
                                                     torch.rand_like(psd_n.real))
                    psd_n = psd_n/10e+10
                    psd_s = psd_s/10e+10
                    psd_n += reg_coeff_tensor
                    psd_n_i = psd_n.inverse()

    # numerator: (..., C_1, C_2) x (..., C_2, C_3) -> (..., C_1, C_3)
    numerator = FC.einsum('...ec,...cd->...ed', [psd_n_i, psd_s])
    # ws: (..., C, C) / (...,) -> (..., C, C)
    ws = numerator / (FC.trace(numerator)[..., None, None] + eps)
    # h: (..., F, C_1, C_2) x (..., C_2) -> (..., F, C_1)
    beamform_vector = FC.einsum('...fec,...c->...fe', [ws, reference_vector])
    return beamform_vector


def get_lcmv_vector(psd_n: ComplexTensor,
                    sv: ComplexTensor,
                    u: torch.Tensor,
                    u2: torch.Tensor,
                    eps: float = 1e-10) -> ComplexTensor:
    C = psd_n.size(-1)
    eye = torch.eye(C, dtype=psd_n.dtype, device=psd_n.device)
    shape = [1 for _ in range(psd_n.dim() - 2)] + [C, C]
    eye = eye.view(*shape)
    psd_n += eps * eye
    try:
        psd_n_i = psd_n.inverse()
    except:
        try:
            reg_coeff_tensor = ComplexTensor(torch.rand_like(psd_n.real),
                                             torch.rand_like(psd_n.real))*1e-2
            psd_n = psd_n/10e+4
            psd_n += reg_coeff_tensor
            psd_n_i = psd_n.inverse()
        except:
            reg_coeff_tensor = ComplexTensor(torch.rand_like(psd_n.real),
                                             torch.rand_like(psd_n.real))*1e-1
            psd_n = psd_n/10e+10
            psd_n += reg_coeff_tensor
            psd_n_i = psd_n.inverse()

    fact1 = FC.einsum('...ec,...cd->...ed', [psd_n_i, sv])
    fact2 = FC.einsum('...ec,...cd->...ed', [sv.conj().transpose(3,2), psd_n_i])
    fact3 = FC.einsum('...ec,...cd->...ed', [fact2, sv])
    C = fact3.size(-1)
    eye = torch.eye(C, dtype=psd_n.dtype, device=psd_n.device)
    shape = [1 for _ in range(psd_n.dim() - 2)] + [C, C]
    eye = eye.view(*shape)
    fact3 += eps * eye
    try:
        fact3_i = fact3.inverse()
    except:
        reg_coeff_tensor = ComplexTensor(torch.rand_like(fact3.real),
                                         torch.rand_like(fact3.real))*1e-2
        fact1 = fact1/10e+4
        fact3 = fact3/10e+4
        fact3 += reg_coeff_tensor
        fact3_i = fact3.inverse()
    ws = FC.einsum('...ec,...cd->...ed', [fact1, fact3_i])
    beamform_vector = FC.einsum('...fec,...c->...fe', [ws, u])
    beamform_vector_2 = FC.einsum('...fec,...c->...fe', [ws, u2])

    return beamform_vector,  beamform_vector_2


def get_gdr_vector(psd_s: ComplexTensor,
                   psd_n: ComplexTensor,
                   sv: ComplexTensor,
                   reference_vector: torch.Tensor,
                   sigma: torch.Tensor,
                   u: torch.Tensor,
                   eps: float = 1e-10) -> ComplexTensor:
    C = psd_n.size(-1)
    eye = torch.eye(C, dtype=psd_n.dtype, device=psd_n.device)
    shape = [1 for _ in range(psd_n.dim() - 2)] + [C, C]
    eye = eye.view(*shape)
    psd_n += eps * eye
    try:
        psd_n_i = psd_n.inverse()
    except:
        try:
            reg_coeff_tensor = ComplexTensor(torch.rand_like(psd_n.real),
                                             torch.rand_like(psd_n.real))*1e-2
            psd_n = psd_n/10e+4
            psd_s = psd_s/10e+4
            psd_n += reg_coeff_tensor
            psd_n_i = psd_n.inverse()
        except:
            try:
                reg_coeff_tensor = ComplexTensor(torch.rand_like(psd_n.real),
                                                 torch.rand_like(psd_n.real))*1e-1
                psd_n = psd_n/10e+10
                psd_s = psd_s/10e+10
                psd_n += reg_coeff_tensor
                psd_n_i = psd_n.inverse()
            except:
                try:
                    reg_coeff_tensor = ComplexTensor(torch.rand_like(psd_n.real),
                                                     torch.rand_like(psd_n.real))*1e-1
                    psd_n = psd_n/10e+10
                    psd_s = psd_s/10e+10
                    psd_n += reg_coeff_tensor
                    psd_n_i = psd_n.inverse()
                except:
                    reg_coeff_tensor = ComplexTensor(torch.rand_like(psd_n.real),
                                                     torch.rand_like(psd_n.real))
                    psd_n = psd_n/10e+10
                    psd_s = psd_s/10e+10
                    psd_n += reg_coeff_tensor
                    psd_n_i = psd_n.inverse()

    # numerator: (..., C_1, C_2) x (..., C_2, C_3) -> (..., C_1, C_3)
    numerator = FC.einsum('...ec,...cd->...ed', [psd_n_i, psd_s])
    # ws: (..., C, C) / (...,) -> (..., C, C)
    ws = numerator / (FC.trace(numerator)[..., None, None] + eps)
    # h: (..., F, C_1, C_2) x (..., C_2) -> (..., F, C_1)
    beamform_vector_1 = FC.einsum('...fec,...c->...fe', [ws, reference_vector])

    fact1 = FC.einsum('...ec,...cd->...ed', [psd_n_i, sv])
    fact2 = FC.einsum('...ec,...cd->...ed', [sv.conj().transpose(3,2), psd_n_i])
    fact3 = FC.einsum('...ec,...cd->...ed', [fact2, sv])
    C = fact3.size(-1)
    eye = torch.eye(C, dtype=psd_n.dtype, device=psd_n.device)
    shape = [1 for _ in range(psd_n.dim() - 2)] + [C, C]
    eye = eye.view(*shape)
    fact3 += eps * eye
    try:
        fact3_i = fact3.inverse()
    except:
        reg_coeff_tensor = ComplexTensor(torch.rand_like(fact3.real),
                                         torch.rand_like(fact3.real))*1e-2
        fact1 = fact1/10e+4
        fact3 = fact3/10e+4
        fact3 += reg_coeff_tensor
        fact3_i = fact3.inverse()
    ws_2 = FC.einsum('...ec,...cd->...ed', [fact1, fact3_i])
    beamform_vector_2 = FC.einsum('...fec,...c->...fe', [ws_2, u])
    sigma = sigma.unsqueeze(-1).expand_as(beamform_vector_1.real)
    one_minus_sigma = torch.ones_like(sigma) - sigma

    beamform_vector = FC.einsum('bfc,bfc->bfc', [sigma, beamform_vector_1]) + FC.einsum('bfc,bfc->bfc', [one_minus_sigma, beamform_vector_2])

    return beamform_vector


def apply_beamforming_vector(beamform_vector: ComplexTensor,
                             mix: ComplexTensor) -> ComplexTensor:
    # (..., C) x (..., C, T) -> (..., T)
    es = FC.einsum('...c,...ct->...t', [beamform_vector.conj(), mix])
    return es
