## Introduction

Many disciplines have been tasked with decoding electroencephalography (EEG) signals, particularly across neuroscience, computer science, and electrical engineering. These communities have developed a variety of paradigms that map multichannel EEG into cognitive, motor, and affective states, enabling applications such as brain-computer interfaces (BCIs), emotion recognition, and seizure prevention.¹ ⁵ ⁹ Deep learning is the dominant approach for accurately classifying EEG signals, particularly using Convolutional Neural Networks (CNNs), Spiking Neural Networks (SNNs) and, more recently, transformer-based models.¹ ⁹ ⁵ In parallel, Large Language Models (LLMs) have demonstrated that deep learning, and particularly transformer models like ChatGPT, can achieve high accuracy across diverse domains and query sizes. However, state-of-the-art LLMs often rely on hundreds of billions of parameters and correspondingly high energy and storage footprints. The tradeoff of energy and accuracy has spurred immense research improving time, space, and energy efficiency via structured sparsity, mixed-precision and post-training quantization, and low-rank decompositions (LRDs) of weight tensors.² ⁴ ¹¹ ⁶ ¹⁰ These techniques exploit that many parameters are redundant or can be represented with lower precision with minimal performance degradation.

EEG decoding faces a related, albeit distinct, set of constraints. Signals are low-SNR, highly non-stationary, and usually available in small datasets; meanwhile, many target applications (wearables, implanted, or bedside systems) operate under computation and energy constraints.¹ ⁸ As a result, there is a growing interest in models that simultaneously maintain decoding accuracy, improve interpretability, and reduce computational cost.¹ ⁸ Recent work includes quantization-aware training for affective state, sparse Bayesian learning (SBL) that combines low-rank spatiotemporal filters with deliberate sparsity, and low-rank matrix and tensor factorization that compress deep networks.¹² ⁷ ³ ⁶ ¹⁰ However, despite these advances, there is little systemic, side-by-side evaluation of these paradigms within a unified EEG decoding setting.¹ ⁸ 

This research proposes a comparative study of three families of methods for energy-aware EEG decoding: (1) quantization and mixed precision approaches, (2) low-rank matrix decompositions, and (3) Bayesian or statistical sparsity-inducing learning. This research will implement simple but representative models instantiating each paradigm on common EEG classification tasks (with an emphasis on affective and BCI-relevant paradigms), define shared evaluation metrics that capture decoding performance and computational efficiency, and analyze the resulting trade-offs. The research goal is to formulate practical recommendations about which techniques are most promising for energy sophistication alongside fact-checking against relevant literature.

## References

¹ Craik et al., “Deep learning for electroencephalogram (EEG) classification tasks: a review,” J Neural Eng., 2019.

² Farina et al., “Sparsity in transformers: A systematic literature review,” Neurocomputing, 2024.

³ Gao et al., “Emotion recognition from multichannel EEG signals based on low-rank subspace self-representation features,” Biomed. Signal Process. Control, 2025.

⁴ Rakka et al., “A Review of State-of-the-art Mixed-Precision Neural Network Frameworks,” IEEE TPAMI, 2024.

⁵ Song et al., “EEG Conformer: Convolutional Transformer for EEG Decoding and Visualization,” IEEE TNSRE, 2022.

⁶ Swaminathan et al., “Sparse low rank factorization for deep neural network compression,” Neurocomputing, 2020.

⁷ Wang et al., “Sparse Bayesian Learning for End-to-End EEG Decoding,” IEEE TPAMI, 2023.

⁸ Xie and Oniga, “A Comprehensive Review of Hardware Acceleration Techniques and Convolutional Neural Networks for EEG Signals,” Sensors, 2024.

⁹ Yan et al., “EEG classification with spiking neural network: Smaller, better, more energy efficient,” Smart Health, 2022.

¹⁰ Yang and Liu, “Stable Low-Rank CP Decomposition for Compression of Convolutional Neural Networks Based on Sensitivity,” Applied Sciences, 2024.

¹¹ Zhang et al., “Post-training Quantization for Neural Networks with Provable Guarantees,” SIAM JMDS, 2023.

¹² Zhong et al., “Knowledge-guided quantization-aware training for EEG-based emotion recognition,” J. Vis. Commun. Image Represent., 2025.

