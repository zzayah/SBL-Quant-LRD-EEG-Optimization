## Abstract

Many disciplines—across neuroscience, computer science, and electrical engineering—are tasked with decoding electroencephalography (EEG) signals. These communities have developed paradigms that map multichannel EEG into cognitive, motor, and affective states, enabling applications such as brain–computer interfaces (BCIs), emotion recognition, and seizure prevention.¹⁵⁹ Deep learning is the dominant approach for accurate EEG classification, particularly using Convolutional Neural Networks (CNNs), Spiking Neural Networks (SNNs), and, more recently, transformer-based models.¹⁹⁵ In parallel, Large Language Models (LLMs) have demonstrated that transformer architectures can achieve high accuracy across diverse domains and query sizes, though state-of-the-art LLMs often rely on hundreds of billions of parameters and incur substantial energy and storage costs. The resulting trade-off between accuracy and energy has spurred substantial research into improving time, space, and energy efficiency via structured sparsity, mixed-precision and post-training quantization, and low-rank decompositions (LRDs) of weight tensors.²⁴¹¹⁶¹⁰ These techniques exploit parameter redundancy, enabling lower-precision or compressed representations with minimal performance degradation.

EEG decoding faces a related, yet distinct, set of constraints. EEG signals are low-SNR, highly non-stationary, and typically available only in small datasets, while many target applications (wearable, implanted, or bedside systems) operate under strict computational and energy limitations.¹⁸ As a result, there is growing interest in models that maintain decoding accuracy while improving interpretability and reducing computational cost.¹⁸ Recent work explores quantization-aware training, Sparse Bayesian Learning (SBL) frameworks that combine low-rank spatiotemporal filtering with deliberate sparsity, and low-rank matrix and tensor factorization approaches for network compression.¹²⁷³⁶¹⁰ However, there is limited systematic, side-by-side evaluation of these paradigms within a unified EEG decoding setting.¹⁸

This work proposes a comparative study of three families of methods for energy-aware EEG decoding:  
1. Quantization and mixed-precision approaches
2. Low-rank matrix decompositions
3. Bayesian / sparsity-inducing learning  

We implement representative models for each paradigm on common EEG classification tasks (with emphasis on affective and BCI-relevant paradigms), define shared evaluation metrics capturing both decoding performance and computational efficiency, and analyze trade-offs. The goal is to provide practical recommendations for balancing accuracy, efficiency, and interpretability.

---

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
