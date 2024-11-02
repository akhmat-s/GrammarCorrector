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
