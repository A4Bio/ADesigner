# Cross-Gate MLP with Protein Complex Invariant Embedding is A One-Shot Antibody Designer

This repository contains the code for the paper "Boosting the Power of Small Multimodal Reasoning Models to Match Larger Models with Self-Consistency Training". Our work focuses on enhancing the capabilities of smaller multimodal reasoning models to achieve performance comparable to larger models.

## Abstract

Antibodies are crucial proteins produced by the immune system in response to foreign substances or antigens. The specificity of an antibody is determined by its complementarity-determining regions (CDRs), which are located in the variable domains of the antibody chains and form the antigen-binding site. Previous studies have utilized complex techniques to generate CDRs, but they suffer from inadequate geometric modeling. Moreover, the common iterative refinement strategies lead to an inefficient inference. In this paper, we propose a \textit{simple yet effective} model that can co-design 1D sequences and 3D structures of CDRs in a one-shot manner. To achieve this, we decouple the antibody CDR design problem into two stages: (i) geometric modeling of protein complex structures and (ii) sequence-structure co-learning. We develop a novel macromolecular structure invariant embedding, typically for protein complexes, that captures both intra- and inter-component interactions among the backbone atoms, including C$\alpha$, N, C, and O atoms, to achieve comprehensive geometric modeling. Then, we introduce a simple cross-gate MLP for sequence-structure co-learning, allowing sequence and structure representations to implicitly refine each other. This enables our model to design desired sequences and structures in a one-shot manner. Extensive experiments are conducted to evaluate our results at both the sequence and structure levels, which demonstrate that our model achieves superior performance compared to the state-of-the-art antibody CDR design methods.

## Acknowledgements 

We highly thank "MEAN: Conditional Antibody Design as 3D Equivariant Graph Translation". [[paper](https://arxiv.org/abs/2208.06073), [code](https://github.com/THUNLP-MT/MEAN/tree/main)]

## Reference
```
@article{tan2024cross,
  title={Cross-Gate MLP with Protein Complex Invariant Embedding is A One-Shot Antibody Designer},
  author={Tan, Cheng and Gao, Zhangyang and Wu, Lirong and Xia, Jun and Zheng, Jiangbin and Yang, Xihong and Liu, Yue and Hu, Bozhen and Li, Stan Z},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2024}
}
```

## Contact

Feel free to contact us through email if you have any questions: Cheng Tan (tancheng@westlake.edu.cn), Westlake University & Zhejiang University.