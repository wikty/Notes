[Project Homepage](https://github.com/allenai/writing-code-for-nlp-research-emnlp2018)

[TOC]



# Two modes of writing research code

* Prototyping
* Writing Components

## Prototyping New Models

Main goals during prototyping

- Write code quickly
- Run experiments, keep track of what you tried
- Analyze model behavior - did it do what you wanted?

### Write code quickly

#### Use a framework

- Training loop?

- Tensorboard logging?
- Model checkpointing?
- Complex data processing, with smart batching?
- Computing span representations?
- Bi-directional attention matrices?

Don’t start from scratch! Use someone else’s components.

#### Get a good starting place

- First step: get a baseline running
- Could be someone else’s code... as long as you can read it
- Even better if this code already modularizes what you want to change
- Re-implementing a SOTA baseline is incredibly helpful for understanding what’s going on, and where some decisions might have been made better

#### Copy first, refactor later

We’re prototyping! Just go fast and find something that works, then go back and refactor (if you made something useful).

- Really bad idea: using inheritance to share code for related models
- Instead: just copy the code, figure out how to share later, if it makes sense

#### Minimal testing (but not no testing)

#### How much to hard-code?

### Run experiments, keep track of what you tried

#### Keep track of what you ran

- You run a lot of stuff when you’re prototyping, it can be hard to keep
  track of what happened when, and with what code

- Currently in invite-only alpha; public beta coming soon
- https://github.com/allenai/beaker
- https://beaker-pub.allenai.org

#### Controlled experiments

- Not good: modifying code to run different variants; hard to keep
  track of what you ran
- Better: configuration files, or separate scripts, or something

### Analyze model behavior

#### Tensorboard

- Crucial tool for understanding model behavior during training
- There is no better visualizer. If you don’t use this, start now.

#### Look at your data!

## Reusable Components

### Abstractions for NLP in the AllenNLP

What are the right abstractions for NLP in the [AllenNLP](https://allennlp.org/)?

Things That We Use A Lot:

- training a model
- mapping words (or characters, or labels) to indexes
- summarizing a sequence of tensors with a single tensor

Things That Have Many Variations:

- turning a word (or a character, or a label) into a tensor
- summarizing a sequence of tensors with a single tensor
- transforming a sequence of tensors into a sequence of tensors

Things that reflect our higher-level thinking:

- we'll have some inputs:
  - text, almost certainly
  - tags/labels, often
  - spans, sometimes
- we need some ways of embedding them as tensors
    - one hot encoding
    - low-dimensional embeddings
- we need some ways of dealing with sequences of tensors
    - sequence in -> sequence out (e.g. all outputs of an LSTM)
    - sequence in -> tensor out (e.g. last output of an LSTM)

### Reusable Components in AllenNLP
AllenNLP is built on PyTorch and is inspired by the question "what higher-level components would help NLP researchers do their research better + more easily?". 

Under the covers, every piece of a model is a torch.nn.Module and every number is part of a torch.Tensor. But we want you to be able to reason at a higher level most of the time.

#### The Model

```python
class Model(torch.nn.Module, Registrable):
	def __init__(self,
				 vocab: Vocabulary,
				 regularizer: RegularizerApplicator = None) -> None: ...

    def forward(self, *inputs) -> Dict[str, torch.Tensor]: ...

    def get_metrics(self, reset: bool = False) -> Dict[str, float]: ...

    @classmethod
	def load(cls,
			 config: Params,
			 serialization_dir: str,
			 weights_file: str = None,
			 cuda_device: int = -1) -> 'Model': ...
```

Model is a subclass of torch.nn.Module

● so if you give it members that are torch.nn.Parameters or are
themselves torch.nn.Modules, all the optimization will just work*
● for reasons we'll see in a bit, we'll also inject any model component
that we might want to configure
● and AllenNLP provides NLP / deep-learning abstractions that allow
us not to reinvent the wheel

#### Model.forward

● returns a dict [!]
● by convention, "loss" tensor is what the training loop will optimize
● but as a dict entry, "loss" is completely optional
○ which is good, since at inference / prediction time you don't have one
● can also return predictions, model internals, or any other outputs
you'd want in an output dataset or a demo

#### Vocabulary

every NLP project needs a Vocabulary

```python
class Vocabulary(Registrable):
    
	def __init__(self,
				 counter: Dict[str, Dict[str, int]] = None,
				 min_count: Dict[str, int] = None,
				 max_vocab_size: Union[int, Dict[str, int]] = None,
				 non_padded_namespaces: Iterable[str] = DEFAULT_NON_PADDED_NAMESPACES,
			     pretrained_files: Optional[Dict[str, str]] = None,
                 only_include_pretrained_words: bool = False,
                 tokens_to_add: Dict[str, List[str]] = None,
                 min_pretrained_embeddings: Dict[str, int] = None) -> None: ...

    @classmethod
	def from_instances(cls, instances: Iterable['Instance'], ...) -> 'Vocabulary': ...

    def add_token_to_namespace(self, token: str, namespace: str = 'tokens') -> int: ...

    def get_token_index(self, token: str, namespace: str = 'tokens') -> int: ...

    def get_token_from_index(self, index: int, namespace: str = 'tokens') -> str: ...
		return self._index_to_token[namespace][index]

    def get_vocab_size(self, namespace: str = 'tokens') -> int: ...
		return len(self._token_to_index[namespace])
```

#### Instances

a Vocabulary is built from Instances

```python
class Instance(Mapping[str, Field]):
    
    def __init__(self, fields: MutableMapping[str, Field]) -> None: ...
    
    def add_field(self, field_name: str, field: Field, vocab: Vocabulary = None) -> None: ...
    
    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]): ...
    
    def index_fields(self, vocab: Vocabulary) -> None: ...
    
    def get_padding_lengths(self) -> Dict[str, Dict[str, int]]: ...
    
    def as_tensor_dict(self,
    				   padding_lengths: Dict[str, Dict[str, int]] = None) -> Dict[str, DataArray]:
```

#### Fields

an Instance is a collection of Fields and a Field contains a data element and knows how to turn it into a tensor

```python
class Field(Generic[DataArray]):

    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]): ...
    
    def index(self, vocab: Vocabulary): ...
    
    def get_padding_lengths(self) -> Dict[str, int]: ...
    
    def as_tensor(self, padding_lengths: Dict[str, int]) -> DataArray: ...
    
    def empty_field(self) -> 'Field': ...
    
    def batch_tensors(self, tensor_list: List[DataArray]) -> DataArray: ...
```

Many kinds of Fields:

● TextField: represents a sentence, or a paragraph, or a question, or ...
● LabelField: represents a single label (e.g. "entailment" or "sentiment")
● SequenceLabelField: represents the labels for a sequence (e.g.
part-of-speech tags)
● SpanField: represents a span (start, end)
● IndexField: represents a single integer index
● ListField[T]: for repeated fields
● MetadataField: represents anything (but not tensorizable)

#### TokenIndexer

● how to represent text in our model is one of the fundamental
decisions in doing NLP
● many ways, but pretty much always want to turn text into indices
● many choices
○ sequence of unique token_ids (or id for OOV) from a vocabulary
○ sequence of sequence of character_ids
○ sequence of ids representing byte-pairs / word pieces
○ sequence of pos_tag_ids
● might want to use several
● this is (deliberately) independent of the choice about how to embed
these as tensors

#### TokenEmbeder

● turns ids (the outputs of your TokenIndexers) into tensors
● many options:
○ learned word embeddings
○ pretrained word embeddings
○ contextual embeddings (e.g. ELMo)
○ character embeddings + Seq2VecEncoder

#### DatasetReader

● "given a path [usually but not necessarily to a file], produce
Instances"
● decouples your modeling code from your data-on-disk format
● two pieces:
○ text_to_instance: creates an instance from named inputs ("passage", "question",
"label", etc..)
○ read: parses data from a file and (typically) hands it to text_to_instance
● new dataset -> create a new DatasetReader (not too much code),
but keep the model as-is
● same dataset, new model -> just re-use the DatasetReader
● default is to read all instances into memory, but base class handles
laziness if you want it

#### DatasetIterator

DatasetIterator handles batching.

● BasicIterator just shuffles (optionally)
and produces fixed-size batches
● BucketIterator groups together
instances with similar "length" to
minimize padding
● (Correctly padding and sorting instances
that contain a variety of fields is slightly
tricky; a lot of the API here is designed
around getting this right)
● Maybe someday we'll have a working
AdaptiveIterator that creates variable
GPU-sized batches

#### Tokenizer

● Single abstraction for both word-level
and character-level tokenization
● Possibly this wasn't the right decision!
● Pros:
○ easy to switch between words-as-tokens
and characters-as-tokens in the same
model
● Cons:
○ non-standard names + extra complexity
○ doesn’t seem to get used this way at all

#### Seq2VecEncoder

from sequence (batch_size, sequence_length, embedding_dim) to vector (batch_size, embedding_dim)

● bag of words
● (last output of) LSTM
● CNN + pooling

#### Seq2SeqEncoder

from sequence (batch_size, sequence_length, embedding_dim) to sequence (batch_size, sequence_length, embedding_dim)

● LSTM (and friends)
● self-attention
● do-nothing

#### Two Different Abstractions for RNNs

● Conceptually, RNN-for-Seq2Seq is
different from RNN-for-Seq2Vec
● In particular, the class of possible
replacements for the former is
different from the class of
replacements for the latter
● That is, "RNN" is not the right
abstraction for NLP!

#### Attention

from (batch_size, sequence_length, embedding_dim), (batch_size, embedding_dim) to (batch_size, sequence_length)

● dot product (xTy)
● bilinear (xTWy)
● linear ([x;y;x*y;...]Tw)

#### MatrixAttention

from (batch_size, sequence_length1, embedding_dim), (batch_size, sequence_length2, embedding_dim) to  (batch_size, sequence_length1, sequence_length2)

● dot product (xTy)
● bilinear (xTWy)
● linear ([x;y;x*y;...]Tw)

#### Attention and MatrixAttention

● These look similar - you could imagine sharing the similarity
computation code
● We did this at first - code sharing, yay!
● But it was very memory inefficient - code sharing isn’t always a good
idea
● You could also imagine having a single Attention abstraction that
also works for attention matrices
● But then you have a muddied and confusing input/output spec
● So, again, more duplicated (or at least very similar) code, but in this
case that’s probably the right decision, especially for efficiency

#### SpanExtractor

● Many modern NLP models use representations of spans of text
○ Used by the Constituency Parser and the Co-reference model in AllenNLP
○ We generalised this after needing it again to implement the Constituency Parser.
● Lots of ways to represent a span:
○ Difference of endpoints
○ Concatenation of endpoints (etc)
○ Attention over intermediate words

### The pipeline in AllenNLP

This seems like a lot of abstractions!

● But in most cases it's pretty simple:
○ create a DatasetReader that generates the Instances you want
■ (if you're using a standard dataset, likely one already exists)
○ create a Model that turns Instances into predictions and a loss
■ use off-the-shelf components => can often write little code
○ create a JSON config and use the AllenNLP training code
○ (and also often a Predictor, coming up next)



# Developing Good Processes

## Source Control

- makes it easy to safely experiment with
  code changes
- makes it easy to collaborate
- makes it easy to revisit older versions of your code
- makes it easy to implement code reviews

## Code Reviews

- code reviewers find mistakes
- code reviewers point out improvements
- code reviewers force you to make your code readable
- code reviewers can be your scapegoat when it turns out your results
  are wrong because of a bug

## Continuous Integration + Build Automation

Continuous Integration: always be merging (into a branch)
Build Automation: always be running your tests (+ other checks)

## Testing

### Write Unit Tests

a unit test is an automated check that a small part of your code works correctly

### What should I test?

- If You're Prototyping, Test the Basics
- If You're Writing Reusable Components, Test Everything
  - test your model can train, save, and load
  - test that it's computing / backpropagating gradients

### How to write tests?

Use test fixtures to write test:

- create tiny datasets that look like the real thing
- use them to create tiny pretrained models
- write unit tests that use them to run your data pipelines and models
  - detect logic errors
  - detect malformed outputs
  - detect incorrect outputs