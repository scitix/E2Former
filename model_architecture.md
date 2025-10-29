## Model Architecture

Efficiently Scaled Attention Interatomic Potential (EScAIP) is a NNIP architecture that leverages highly optimized self-attention mechanisms for scalability and expressivity.

![Illustration of the Efficient Graph Attention Potential model architecture. The model consists of $B$ graph attention blocks (dashed box), each of which contains a geometric-aware graph attention layer, a feed forward layer, and two readout layers for node and edge features. The concatenated readout from each block are used to predict per-atom forces and system energy.](assets/model_diag.jpg)


The core design of our EScAIP model is centered around scalability and efficiency. To avoid costly tensor products, we operate on scalar features that are invariant to rotations and translations. This enables us to take advantage of the highly optimized self-attention mechanism from natural language processing, making the model substantially more time and memory efficient than models that use equivariant group representations. We describe the key components of the model and the motivation behind their design:

### Input Block

The input to the model is a radius-$r$ graph representation of the molecular system. We use three attributes from the molecular graph as input: atomic numbers, Radial Basis Expansion (RBF) of pairwise distances, and Bond Order Orientation (BOO) features (see Appendix A). The atomic numbers embeddings are used to encode the atom type information, while the RBF and BOO embeddings are used to encode the spatial information of the molecular system. The input features are then passed through a feed forward neural network (FFN) to produce the initial edge and node features.

### Graph Attention Block

The core component of the model is the graph attention block. It takes node features and molecular graph attributes as input. All the features are projected and concatenated into a large message tensor of shape $(N, M, H)$, where $N$ is the number of nodes, $M$ is the max number of neighbor, and $H$ is the feature dimension. The message tensor is then processed by a multi-head self-attention mechanism. The attention is parallelized over each neighborhood, where $M$ is the sequence length. There is no positional encoding to ensure permutation equivariance. By using customized Trition kernels, the attention mechanism is highly optimized for GPU acceleration. The output of the attention mechanism is then aggregated back to the atom level. The aggregated messages are then passed through the Feed Forward Network to produce the output node features.

### Readout Block

Following GemNet-OC, we use two readout layers for each graph attention block. The first readout layer takes in the unaggregate messages from the graph attention block and produces edge readout features. The second readout layer takes in the output node features from the FFN and produces node readout features. The node and edge readout features from all graph attention blocks are concatenated and passed into the output block for output prediction.

### Output Block

The output block takes the concatenated readout features and predicts the per-atom forces and system energy. The energy prediction is done by an FFN on the node readout features. The force prediction is divided into two parts: the force magnitude is predicted by an FFN on the node readout features, and the force direction is predicted by a transformation of the unit edge directions with an FFN on the edge readout features. As opposed to GemNet, the transformation is not scalar but vector-valued. Thus, the predicted force direction is not equivariant to rotation of the input data. In our experiments, we found this symmetry-breaking output block made the model perform better. The reason could be that this formulation is more expressive and easier to optimize.

We also note that predicting the force magnitude from node readout features is very helpful for energy prediction. This is because the energy prediction is a global property of the molecular system, while the force magnitude is a local property of the atom. By guiding the model towards a fine-grained force magnitude prediction, the model can learn a better representation of the molecular system, which can help it to predict the system energy more accurately.

## Appendix A: Bond-Orientational Order Feature

A central question for NNIP models is how to incorporate bond directional information. As opposed to other domains, such as social networks, the edges (or bonds) in molecular graphs possess distinct geometric attributes, i.e., pairwise directions. However, the raw value of the bond direction changes with the rotation and translation of the molecule, making it challenging to directly utilize these features in NNIP models.

Group representation models, such as NequIP, handle this by expanding bond directions using spherical harmonics, and use this in the tensor product computation to incorporate the directional information into the node features. Since both the spherical harmonics and tensor product operation are equivariant to rotations, it ensures that the node features are rotationally equivariant. However, this approach is computationally inefficient, especially with higher rotation orders.

Conversely, models that use scalar features, such as GemNet, use heavily engineered basis expansions of the bond directional and angular information as edge features. These features are rotationally invariant, making the node features invariant to rotations. However, these expansions lack flexibility and expressivity, which can be limiting for large-scale datasets.

To this end, we propose a simple and data-driven approach to embed the bond directional information. To avoid the computational inefficiency of taking a tensor product, we aim to use the simplest possible representation of bond direction that is rotationally invariant. Inspired by [steinhardt1983](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.28.784), we use an embedding of the Bond-Orientational Order (BOO) to represent the directional features. Formally, for a node $v$, the BOO of order $l$ is,

$\mathrm{BOO}^{(l)}(v)=\sum\_{m=-l}^{l}\sqrt{\frac{4\pi}{2l+1}\left|\frac{1}{n_v}\sum\_{u\in \mathrm{Nei}(v)}Y^{(l)}\_{m}(\hat{\boldsymbol{d}}\_{uv})\right|^2}$

$\mathrm{BOO}(v) = \mathrm{Concat}\left(\{\mathrm{BOO}^{(l)}(v)\}\_{l=0}^{L}\right),$

where $\hat{\boldsymbol{d}}\_{uv}$ is the normalized bond direction vector between node $v$ and $u$, $n_v$ is the number of neighbors of $v$, $\mathrm{Nei}(v)$ is the neighbors of $v$, $Y^{(l)}\_{m}$ is the spherical harmonics of order $l$ and degree $m$. This can be interpreted as the minimum-order rotation-invariant representation of the $l$-th moment in a multipole expansion for the distribution of bond vectors $\rho_{\rm bond}(\mathrm n)$ across a unit sphere. In other words, BOO is the \textbf{simplest} way to encode the neighborhood directional information in a rotationally invariant manner. The BOO features $\mathrm{BOO}(v)\in \mathbb R^{L+1}$ for a node $v$ is the concatenation of $\mathrm{BOO}(v)^{(l)}$. In theory, the BOO feature contains all the directional information of the neighborhood, and the embedding network can learn to extract such information.
