# Transformer Language Model from Scratch

This project implements a standard Transformer language model entirely from scratch, utilizing only PyTorch's primitive operations. This includes the implementation of all necessary Transformer layers, and training components  such the optimizer, learning rate scheduler and gradient clipping. Additionally, a reasonably fast Byte Pair Encoding (BPE) tokenizer has been included for text processing.

## Implementation Details

The general guidelines for this implementation were followed from [Stanford CS336, assignment 1](https://stanford-cs336.github.io/spring2025/). To ensure the accuracy and correctness of all modules, they were tested against the test files provided in [Stanford CS336, assignment 1](https://stanford-cs336.github.io/spring2025/).

## Dataset and Training

The model was trained on the "TinyStories" dataset, which comprises children's stories.

## Sample Output

Below are sample outputs (generated with temperature scaling) from the trained model:

Prompt: *Once upon a time, in*

Output: *Once upon a time, in a small town, there was a little boy named Tim. Tim loved to play with his toy arrow. One day, he went to the park with his mom to play with his friends.
While playing, Tim saw a big tree. He wanted to climb it. So, he started to climb the tree. When he got to the top, he saw a bird. The bird said, "Hello, Tim! I can play with you!"
Tim was so happy. He played with his friends all day long. They had so much fun. When it was time to go home, Tim said, "I want to come back to the park tomorrow." The bird said, "I can come back tomorrow, and you can play with my friends again."
And so, Tim went home with his big, red arrow, and they all had a great day at the park.
<|endoftext|>*

Prompt: *A brown fox*

Output: *A brown fox. The fox was very hungry. The fox wanted to eat the fox.
The fox saw the fox and was scared. The fox thought, "I will eat the fox!" The fox ran away. The fox ate the fox. The fox was very happy.
But then, something unexpected happened. The fox ate the fox too! The fox was not hungry anymore. The fox was happy and not hungry anymore. The fox and the fox became friends.
<|endoftext|>*
