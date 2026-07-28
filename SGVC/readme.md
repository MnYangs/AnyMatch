# Sample‑level Geometric Consistency Verification (SGCV)

The **SGCV** module is a quality‑control step in the AnyMatch pipeline. After generating a novel‑view image pair, it uses a pretrained dense matching model (RoMa) to compute the **Percentage of Correct Keypoints (PCK)** at a threshold `τ` between the rendered correspondence and the ground‑truth geometric projection.  

A sample is **retained** only if `PCK@τ ≥ η` (e.g., τ = 5 px, η = 0.6); otherwise, it is **filtered out**. This effectively removes hallucinated or geometrically inconsistent samples introduced during inpainting or depth estimation, ensuring that the final multi‑modal dataset provides reliable, physically plausible supervision for training matching networks.

**Start View Transformation**. Please run：

```bash
python ./SGVC/SGVC_pck.py
```

Please note that the input of the SGCV module comes from 
`./View_Transformation/AnyMatch_results/image1` and `./View_Transformation/AnyMatch_results/image2`.