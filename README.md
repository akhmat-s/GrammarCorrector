# GrammarCorrector
This tool fixes grammar and spelling errors in inputted text.

"[FlanT5 from scratch for the grammar correction tool](https://medium.com/@akhmat-s/flant5-from-scratch-for-the-grammar-correction-tool-deadba9a6778)" article about how this models was trained:
>FlanT5 was trained using [JFLEG](https://arxiv.org/abs/1702.04066) dataset. The primary objective of the experiment was to develop a highly effective tool using relatively small models, minimal datasets, and constrained computational resources.
>
>To accomplish this goal, we implemented two key strategies:
>- [Perplexity-Based Data](https://arxiv.org/abs/2405.20541) Pruning With Small Reference Models.
>- A simple sampling and voting method for [multiple LLM agents](https://arxiv.org/abs/2402.05120).

## Installation
```python
pip install -U git+https://github.com/akhmat-s/GrammarCorrector.git
```

## Quick Start
```python

from grammarcorrector import GrammarCorrector

grammar = GrammarCorrector()

examples = [
    "The world depend of us.",
    "Can you believe it I'm finally graduating.",
    "He don't likes vegetables and I don’t like fruit.",
    "We have went to that restaurant many times.",
    "They are meeting  the park on 3 pm",
    "She is knowing the answers to the question.",
    "People are suffering in his own way.",
    "Sometimes you need to listen to yourself although something bad is happen.",
    "You should think carefully to whatever they say and sometimes simple-looking peoply they are nicer and more honest than goodlooking people."
]

for example in examples:
    corrected_sentence = grammar.correct(example)
    print("[Input] ", example)
    print("[Correction] ",corrected_sentence)
    print("-" *100)
```

```text
[Input]  The world depend of us.
[Correction]  The world depends on us.
----------------------------------------------------------------------------------------------------
[Input]  Can you believe it I'm finally graduating.
[Correction]  Can you believe it that I'm finally graduating.
----------------------------------------------------------------------------------------------------
[Input]  He don't likes vegetables and I don’t like fruit.
[Correction]  He doesn't like vegetables, and I don't like fruit.
----------------------------------------------------------------------------------------------------
[Input]  We have went to that restaurant many times.
[Correction]  We have gone to that restaurant many times.
----------------------------------------------------------------------------------------------------
[Input]  They are meeting  the park on 3 pm
[Correction]  They are meeting at the park at 3 pm.
----------------------------------------------------------------------------------------------------
[Input]  She is knowing the answers to the question.
[Correction]  She knows the answers to the question.
----------------------------------------------------------------------------------------------------
[Input]  People are suffering in his own way.
[Correction]  People are suffering in their own way.
----------------------------------------------------------------------------------------------------
[Input]  Sometimes you need to listen to yourself although something bad is happen.
[Correction]  Sometimes you need to listen to yourself, even though something bad happens.
----------------------------------------------------------------------------------------------------
[Input]  You should think carefully to whatever they say and sometimes simple-looking peoply they are nicer and more honest than goodlooking people.
[Correction]  You should think carefully about whatever they say, and sometimes simple-looking people are nicer and more honest than good-looking people.
----------------------------------------------------------------------------------------------------
```
