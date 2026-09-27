Read an einsum string as: letters on the left name the axes of each input (comma-separated), letters on the right name the output axes. Any letter missing from the output is summed over. A letter appearing twice within one input walks its diagonal.
---
`batched_matmul`: `"bij,bjk->bik"`. `trace_of_each`: `"bii->b"`. `diagonal_of_each`: `"bii->bi"`. `outer`: `"i,j->ij"`.
---
`bilinear`: `"bi,ij,bj->b"` (three operands are fine). `attention_scores`: `"bqd,bkd->bqk"`. `row_dot`: `"nd,nd->n"`.
