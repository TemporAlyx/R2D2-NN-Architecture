# R2D2-NN-Architecture
Experiments of neural network architecture(s) utilizing the concepts of Recursive Recontextualization and Dynamic Distillation (R2D2), a collection of ideas meant to explore an alternative structure to contemporary neural network architectures, with a focus on adaptability in both scale and modality.

This project is a work in progress, in part a learning exercise to develop my AI development and research skills the only way I know how: by taking on increasingly ambitious projects and failing in what I'd like to believe is a vaguely upwards direction.

# Core Concepts
Note: the implemented code in this repo has some differences to the below descriptions, explained in the current implementation section


- Recursive Recontextualization
The primary motivation underlying this restructuring of a traditional model is based on the idea of optimal information flow within a network. At a given step in a network, we want to separate information that is not relevant to a given goal, information that requires further processing to be useful, and information that is already a useful representation for the target. In most modern architectures, we utilize skip connections to allow information to flow from early layers of the network to the end, but this may be mixing representations that we want to keep static (target data), and working representations of the data.
From this idea, we instead keep separate structures for the input data, an internal context, and the output target(s). At each step in the neural network, we utilize the context to compute the weights for all the processing within this step, wherein we compute a view of the data, combine it with a view of the context, and then predict an update to the context. Then we compute a view of the target and updated context to predict an update to the target.
Since we are predicting the weights at each step, the model doesn’t have a set number of processing blocks, and we can dynamically adjust how deep we intend to process, recursively developing the context with views from the data in order to refine its prediction. 

Ideally, this allows ‘easy’ target information to be quickly added to the output, while still allowing deeper representations to be developed over multiple recursive passes, with the depth of processing definable by the user to prioritize speed over accuracy. Another potential benefit is that since weights are generated per sample, deeper layers may be able to process in ways specifically tailored to a given input sample.
In practice, recursive structures like this are difficult to backpropagate through, and requiring the model to handle processing while also predicting target updates at each step may require large internal sizes in order to allow for the necessary processing bandwidth. Additionally, as model context is per sample, and thus weights are generated per sample, processing with large batch sizes poses some difficulties.

- Dynamic Distillation
The recursive nature of the model allows for the depth of the model to be variable, and so we can weight the loss function such that it tries to match deeper patterns at earlier steps in the model.
Additionally, we can train subsets of a large model to operate as smaller models, and reinforce the learning in a similar way, where subsets of the internal context width are encouraged to match the predictions of the full width model. 
Ideally, sparse but useful representations are learned within the full width and depth of the model, while at the same time smaller subsets of the model are trained to mimic this, which in turn frees up more space in the full model to develop new representations.

# Current Implementation
The code in this repo is a somewhat simplified version of the above concepts. Rather than separating the targets, the data and targets are represented in the same space, and the model acts as a sort of autoencoder. Data type is added into the processing as an embedding similar to the positional encoding, and for the test case in the notebook (imagenet classification), there is a type associated with the target for prediction. Additionally, part of the model is designed to  learn the associated inverse step as a measure to potentially improve overall learning, but whether this works (and whether the whole collection of ideas work for that matter), is still to be determined. 
A major difficulty has been vram limitations, and, adjacently, the difficulty with implementing batching due to the sample specific weight generation. 

# Future Plans
- It may be a good idea to reduce the overall scope (ya think?), just a bit, by splitting off some of the ideas. Regardless of whether they become separate projects, the many interacting ideas here need to be otherwise redesigned such that it is easier to isolate and test how they function independently.
- The encoding and decoding structures are in need of being rewritten with standardized input padding or something to the effect such that tensorflow graph execution is significantly easier.
- Many of the issues facing smooth testing of this idea revolve around the problems introduced with a recursive layer architecture, and some of the problems are not dissimilar to problems facing Recurrent Neural Networks, and better understanding recent RNN advances may help. (such as https://github.com/BlinkDL/RWKV-LM)
