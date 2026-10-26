"""Categorical Disagreement Metric (D3).

Evaluates hard decision-level disagreement based on argmax predictions:
- 0 = all 3 modalities agree (yt == ya == yv)
- 1 = exactly two modalities agree and one differs
- 2 = all three modalities differ (yt != ya != yv != yt)
"""

from typing import Union
import numpy as np
import torch


def compute_categorical_disagreement_d3(
    pred_text: Union[int, np.ndarray, torch.Tensor],
    pred_audio: Union[int, np.ndarray, torch.Tensor],
    pred_visual: Union[int, np.ndarray, torch.Tensor],
) -> Union[int, np.ndarray, torch.Tensor]:
    """Compute D3 categorical disagreement level from modality predictions or distributions.

    Args:
        pred_text: Predicted label index or probability distribution (shape: [...] or [..., K])
        pred_audio: Predicted label index or probability distribution (shape: [...] or [..., K])
        pred_visual: Predicted label index or probability distribution (shape: [...] or [..., K])

    Returns:
        Categorical disagreement level: 0, 1, or 2 (scalar or tensor/array).
    """
    if isinstance(pred_text, torch.Tensor):
        if pred_text.ndim > 0 and pred_text.shape[-1] > 1:
            pred_text = torch.argmax(pred_text, dim=-1)
        if pred_audio.ndim > 0 and pred_audio.shape[-1] > 1:
            pred_audio = torch.argmax(pred_audio, dim=-1)
        if pred_visual.ndim > 0 and pred_visual.shape[-1] > 1:
            pred_visual = torch.argmax(pred_visual, dim=-1)

        pred_text = pred_text.to(torch.long)
        pred_audio = pred_audio.to(torch.long)
        pred_visual = pred_visual.to(torch.long)

        eq_ta = (pred_text == pred_audio)
        eq_tv = (pred_text == pred_visual)
        eq_av = (pred_audio == pred_visual)

        # All 3 agree: eq_ta & eq_tv & eq_av
        all_agree = eq_ta & eq_tv & eq_av
        # All 3 differ: (~eq_ta) & (~eq_tv) & (~eq_av)
        all_differ = (~eq_ta) & (~eq_tv) & (~eq_av)

        d3 = torch.ones_like(pred_text, dtype=torch.long) # Default to 1 (one differs)
        d3 = torch.where(all_agree, torch.zeros_like(d3), d3)
        d3 = torch.where(all_differ, torch.full_like(d3, 2), d3)

        if d3.ndim == 0:
            return int(d3.item())
        return d3

    # NumPy / scalar path
    yt = np.asarray(pred_text)
    ya = np.asarray(pred_audio)
    yv = np.asarray(pred_visual)

    if yt.ndim > 0 and yt.shape[-1] > 1:
        yt = np.argmax(yt, axis=-1)
    if ya.ndim > 0 and ya.shape[-1] > 1:
        ya = np.argmax(ya, axis=-1)
    if yv.ndim > 0 and yv.shape[-1] > 1:
        yv = np.argmax(yv, axis=-1)

    eq_ta = (yt == ya)
    eq_tv = (yt == yv)
    eq_av = (ya == yv)

    all_agree = eq_ta & eq_tv & eq_av
    all_differ = (~eq_ta) & (~eq_tv) & (~eq_av)

    d3 = np.ones_like(yt, dtype=np.int64)
    d3 = np.where(all_agree, 0, d3)
    d3 = np.where(all_differ, 2, d3)

    if d3.ndim == 0:
        return int(d3)
    return d3


def d3_to_one_hot(
    d3: Union[int, np.ndarray, torch.Tensor],
    num_levels: int = 3,
) -> Union[np.ndarray, torch.Tensor]:
    """Convert categorical disagreement levels (0, 1, 2) to one-hot vectors."""
    if isinstance(d3, torch.Tensor):
        if d3.ndim == 0:
            d3_t = d3.unsqueeze(0)
            one_hot = torch.nn.functional.one_hot(d3_t.to(torch.long), num_classes=num_levels).to(torch.float32)
            return one_hot.squeeze(0)
        return torch.nn.functional.one_hot(d3.to(torch.long), num_classes=num_levels).to(torch.float32)

    d3_arr = np.asarray(d3, dtype=np.int64)
    if d3_arr.ndim == 0:
        vec = np.zeros(num_levels, dtype=np.float32)
        vec[d3_arr] = 1.0
        return vec
    
    one_hot = np.zeros((*d3_arr.shape, num_levels), dtype=np.float32)
    np.put_along_axis(one_hot, np.expand_dims(d3_arr, axis=-1), 1.0, axis=-1)
    return one_hot
