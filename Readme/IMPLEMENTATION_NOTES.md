# MNCL implementation notes

This project is **paper-first**: equations and module boundaries follow the supplied MNCL paper. The public author repository is used only to resolve details that the paper does not state clearly. The two sources are not fully identical.

## Choices kept from the paper

- The recommendation objective uses BPR as written in Eq. 21 and Eq. 22.
- The multi-negative contrastive loss keeps the `omega` weighting from Eq. 15-16.
- The final fusion follows Eq. 18-19 exactly: user `[e_s, e_s, e_m]`, item `[e_s, e_m, e_g]`.
- Noise direction follows Eq. 6 using the sign of the current layer input.
- AUC and F1 are computed globally over the complete test split.

## Details resolved from the public author code

- Contrastive temperature is set to `tau = 0.6`. The paper defines `tau` but does not report its value; the public code hard-codes `0.6` in its contrastive functions.
- Semantic KNN uses `topk = 10`.
- Structural sampling uses `0.5` and message dropout uses `0.1` as practical defaults.

## Known paper / code differences

1. **BPR vs BCE**  
   The paper states BPR, while the public function named `create_bpr_loss` actually applies sigmoid + binary cross entropy. This project keeps the paper formulation.

2. **Contrastive weighting**  
   The paper explicitly includes `omega`; the public implementation does not expose the same weighting in its shown contrastive denominator. This project keeps `omega`.

3. **Fusion order**  
   The paper and public code concatenate view embeddings in different orders/combinations. This project follows Eq. 18-19.

4. **Noise sign**  
   Eq. 6 uses the current layer representation. The public code applies the sign after graph propagation. This project follows the equation.

5. **Evaluation aggregation**  
   The public training script averages metrics over batches. This project computes one global AUC/F1 over all test examples.

6. **Train/test split**  
   The paper does not specify a precise split algorithm in the supplied text. This project uses a reproducible user-wise, label-stratified `test_ratio=0.2` split.

## Reproduction warning

Because the paper and public implementation differ, reproducing the exact table values is not guaranteed by a paper-first implementation. If exact repository reproduction becomes the goal, add a separate compatibility mode rather than silently changing the paper-first path.
