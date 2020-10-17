# DaisyGANv2
This is a [Generative adversarial network](https://en.wikipedia.org/wiki/Generative_adversarial_network) which digests word arrays.

Following on from DaisyGAN v1 left off; a significant modification was made so that the network would process indexed words form a pre-computed word table.

The word table was generated by taking a number of messages from a random distribution of telegram users, these messages where then sanitised to remove any non-alpha characters, convert uppercase to lowercase characters, and finally split the messages into a list of unique words; this made up the entirety of the word table.

The intention of this word table is that this would be the entire vernacular that the neural net would be capable of understanding and processing, as such, the index of this table was normalised to a -1 to +1 floating-point range before being fed into the neural network. In place of a word being supplied which was not present in the table, a 0 was supplied to the input perceptron at that position in the network, which does actually map to the word "discord" but serves as a "no word / blank" identifier, so yes, woops, one word in the word table will inevitably end up ignored; a loss I can cope with.

The network only recognises words of up to 16 characters and each sentence can only have 16 words maximum.

Unfortunately, the performance of this network was still particularly poor, however notably better than the original.

## Example Usage
- ```./cfdgan retrain <optional file path>``` - You want to start off by training the network from the provided dataset.
- ```./cfdgan "this is an example scentence"``` - Get a percentage of likelyhood that the user Daisy wrote the provided message.
- ```./cfdgan rnd``` - Get the percentage of likelyhood that Daisy wrote a provided random message.
- ```./cfdgan ask``` - A never ending console loop where you get to ask what percentage likelyhood Daisy wrote a given message.
- ```./cfdgan``` - The adversarial message generator, not for the light hearted, it seems to mostly be racist or read off like its generating political porn video descriptions ..
