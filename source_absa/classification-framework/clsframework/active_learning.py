import math
import torch
import numpy as np


def logit_mean(logits, dim: int, keepdim: bool = False):
    r"""Computes $\log \left ( \frac{1}{n} \sum_i p_i \right ) =
    \log \left ( \frac{1}{n} \sum_i e^{\log p_i} \right )$.

    We pass in logits.
    """
    return torch.logsumexp(logits, dim=dim, keepdim=keepdim) - math.log(logits.shape[dim])


def entropy(logits, dim: int, keepdim: bool = False):
    return -torch.sum((torch.exp(logits) * logits).double(), dim=dim, keepdim=keepdim)


def mutual_information(logits_B_K_C):
    """Returns the mutual information between each element of the batch and the model parameter.
    Returns H(y_i) - E_p(w)[H(y_i|w)] for i in B
    """
    # Entropy of every sample -> H(y_i|w_j) for i in B and j in K
    sample_entropies_B_K = entropy(logits_B_K_C, dim=-1)
    # Entropy of one segment, averaged over all MC samples -> E_p(w)[H(y_i|w)] for i in B
    entropy_mean_B = torch.mean(sample_entropies_B_K, dim=1)

    # Logits averaged over all samples (expected value) -> E_p(w)[y_i] for i in B
    logits_mean_B_C = logit_mean(logits_B_K_C, dim=1)
    # Entropy of averaged probs -> H(y_i) for i in B
    mean_entropy_B = entropy(logits_mean_B_C, dim=-1)

    mutual_info_B = mean_entropy_B - entropy_mean_B
    return mutual_info_B


def conditional_entropy_from_logits_B_K_C(logits_B_K_C):
    B, K, C = logits_B_K_C.shape
    return torch.sum(-logits_B_K_C * torch.exp(logits_B_K_C), dim=(1, 2)) / K


def batch_conditional_entropy_B(logits_B_K_C, out_conditional_entropy_B=None):
    B, K, C = logits_B_K_C.shape

    if out_conditional_entropy_B is None:
        out_conditional_entropy_B = torch.empty((B,), dtype=torch.float64)
    else:
        assert out_conditional_entropy_B.shape == (B,)

    for conditional_entropy_b, logits_b_K_C in split_tensors(out_conditional_entropy_B, logits_B_K_C, 8192):
        logits_b_K_C = logits_b_K_C.double()
        conditional_entropy_b.copy_(conditional_entropy_from_logits_B_K_C(logits_b_K_C), non_blocking=True)

    return out_conditional_entropy_B


def gather_expand(data, dim, index):
    # assert len(data.shape) == len(index.shape)
    # ic(list(zip(data.shape, index.shape)))
    # assert all(dr == ir or 1 in (dr, ir) for dr, ir in zip(data.shape, index.shape))

    max_shape = [max(dr, ir) for dr, ir in zip(data.shape, index.shape)]
    new_data_shape = list(max_shape)
    new_data_shape[dim] = data.shape[dim]

    new_index_shape = list(max_shape)
    new_index_shape[dim] = index.shape[dim]

    data = data.expand(new_data_shape)
    index = index.expand(new_index_shape)

    # In the used case dim=-1 so torch.gather results in:
    # out[i,j,k,l] = data[i,j,k,index[i,j,k,l]]
    return torch.gather(data, dim, index)


def batch_multi_choices(probs_b_C, M: int):
    """
    Returns sampled class labels accoding to the given prob. distribution
    """
    probs_B_C = probs_b_C.reshape((-1, probs_b_C.shape[-1]))

    # samples: Ni... x draw_per_xx
    choices = torch.multinomial(probs_B_C, num_samples=M, replacement=True)

    choices_b_M = choices.reshape(list(probs_b_C.shape[:-1]) + [M])
    return choices_b_M


def sample_M_K(probs_B_K_C, S=1000):
    probs_B_K_C = probs_B_K_C.double()

    K = probs_B_K_C.shape[1]

    # Given the probabilities in probs_B_K_C, take S samples from
    # class labels according to the given distributions. In essence,
    # we are sampling a possible configuration of class labels y_1:n
    # for all samples
    choices_N_K_S = batch_multi_choices(probs_B_K_C, S).long()

    # Insert an empty dimension
    expanded_choices_N_K_K_S = choices_N_K_S[:, None, :, :]
    expanded_probs_N_K_K_C = probs_B_K_C[:, :, None, :]

    # From the sampled class labels gather the probabilities of those classes
    probs_N_K_K_S = gather_expand(
        expanded_probs_N_K_K_C,
        dim=-1,
        index=expanded_choices_N_K_K_S,
    )
    # Calculate the probability of all observed class labels
    # exp sum log seems necessary to avoid 0s?
    probs_K_K_S = torch.exp(torch.sum(torch.log(probs_N_K_K_S), dim=0, keepdim=False))
    samples_K_M = probs_K_K_S.reshape((K, -1))

    samples_M_K = samples_K_M.t()
    return samples_M_K


def importance_weighted_entropy_p_b_M_C(p_b_M_C, q_1_M_1, M: int):
    return torch.sum(-torch.log(p_b_M_C) * p_b_M_C / q_1_M_1, dim=(1, 2)) / M


def split_tensors(output, input, chunk_size):
    "Returns output and input tensor splits as tuples"
    assert len(output) == len(input)
    return list(zip(output.split(chunk_size), input.split(chunk_size)))


def batch(probs_B_K_C, samples_M_K):
    # Bring everything to the correct format and to the same device
    probs_B_K_C = probs_B_K_C.double()
    samples_M_K = samples_M_K.double()

    device = probs_B_K_C.device
    M, K = samples_M_K.shape
    B, K_, C = probs_B_K_C.shape
    assert K == K_

    p_B_M_C = torch.empty((B, M, C), dtype=torch.float64, device=device)

    for i in range(B):
        torch.matmul(samples_M_K, probs_B_K_C[i], out=p_B_M_C[i])

    p_B_M_C /= K

    q_1_M_1 = samples_M_K.mean(dim=1, keepdim=True)[None]

    # Now we can compute the entropy.
    # We store it directly on the CPU to save GPU memory.
    entropy_B = torch.zeros((B,), dtype=torch.float64)

    chunk_size = 256
    for entropy_b, p_b_M_C in split_tensors(entropy_B, p_B_M_C, chunk_size):
        entropy_b.copy_(importance_weighted_entropy_p_b_M_C(p_b_M_C, q_1_M_1, M), non_blocking=True)

    return entropy_B


def select_segments(logits_B_K_C, acquisitionsize, preselectionsize, samples, verbose=False):
    # Bring logits_B_K_C into correct form
    logits_B_K_C = torch.tensor(np.stack(logits_B_K_C, axis=1)).log()
    # Calculate mutual information of all segments and model parameters
    scores_B = mutual_information(logits_B_K_C)

    # Select preselectionsize segment aspect pairs with highest mutual information.
    # Others are very unlikely to be useful anyway
    n = preselectionsize
    maxindxs = scores_B.sort().indices[-n + 1:-1]
    logits_B_K_C = logits_B_K_C[maxindxs]
    scores_B = scores_B[maxindxs]
    # segments = [segments[i] for i in maxindxs]

    maxindx2globalidx = {i: maxindxs[i].item() for i in range(len(maxindxs))}

    subset_acquisition_bag = []  # Currently selected samples for labeling

    # Since we have nothing in the acquisition bag, use scores_B as init for a_batchBALD
    partial_multi_bald_B = scores_B

    # Calculate conditional entropies
    # conditional_entropies_B = E_p(w)[H(y_i|w)]. After summing
    # together we get E_p(w)[H(y_1, ..., y_n|w)] which is the right
    # hand side of Equation 8 to calculate batchBALD
    conditional_entropies_B = batch_conditional_entropy_B(logits_B_K_C)

    # Calculate probabilities
    probs_B_K_C = logits_B_K_C.exp()

    scores = []
    # Iteratively add samples to our acquisition bag
    for i in range(acquisitionsize):
        if i != 0:
            # Calculate P^_1:n-1
            samples_M_K = sample_M_K(probs_B_K_C[subset_acquisition_bag], S=samples)
            # Calculate joint entropies H(y_1:n)
            joint_entropies_B = batch(probs_B_K_C, samples_M_K)
            # a_batchBALD without right hand side of acquisition bag
            partial_multi_bald_B = joint_entropies_B - conditional_entropies_B

        # Don't consider segments already in the aquisition bag
        partial_multi_bald_B[subset_acquisition_bag] = -math.inf

        # Decide which sample should be added to the aquisition bag
        winner_index = partial_multi_bald_B.argmax().item()

        meanlog = torch.mean(torch.softmax(logits_B_K_C[winner_index], dim=1), dim=0)
        if verbose:
            print(f"{i+1}/{acquisitionsize} Classprobabilities {[round(j.item(),2) for j in meanlog]},",
                  f"score_B = {scores_B[winner_index]}")
            print(winner_index)
        scores.append(scores_B[winner_index])
        # Add new item to acquisition bag
        subset_acquisition_bag.append(winner_index)
    # Map selected segments back to global segment indices
    return [maxindx2globalidx[i] for i in subset_acquisition_bag], sum(scores) / len(scores)
