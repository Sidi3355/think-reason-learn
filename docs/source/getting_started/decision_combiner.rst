Letting a decision model decide
===============================

Random Rule Forest (RRF) writes yes/no questions, has the answerer answer them for every sample, and then
combines the answers into one prediction. By default an elastic net weighs them (``combiner="elasticnet"``), or a
vote counts them (``combiner="vote"``, the paper's). ``combiner="decision"`` hands that last step to a **decision
model** instead: Jev, Kev on your own server, or any model behind TypeSafe's System One API.

How it works
------------

For each sample, RRF writes its answers as text, one line per rule, best rule first:

.. code-block:: text

   Has the founder founded a company before? yes
   Does the founder have a prior exit? no
   Did the founder study at a top-50 university? unknown

A line ends in ``yes`` when the answer's probability of yes is at least 0.5, ``no`` below that, and ``unknown``
when the answerer gave no answer. The decision model gets that text and one yes/no question, by default whether
the sample is the positive class given your ``task``. Its probability of yes is the sample's score, and ``fit``
tunes the threshold for F-beta on the scored samples, as the other combiners do.

Nothing is fitted inside RRF for this combiner: the decision model decides as it is. All of RRF's questions go to
it (the best ``max_rules`` by F-beta, if set), so it can use answers an elastic net would weigh at zero, and
combine them in ways a linear model can't.

Using it
--------

.. code-block:: python

   from think_reason_learn import LLM
   from think_reason_learn.rrf import RRF

   kev = LLM("kev", base_url="http://127.0.0.1:8009/v1/systemone", kind="decision")

   model = RRF(
       task="Predict whether this startup founder's company will succeed.",
       generator="anthropic:claude-sonnet-5-5",
       answerer="jev",
       combiner="decision",
       combiner_model=kev,  # or "jev"
       combiner_question="Will this founder's company succeed (exit, IPO or funding over $500M)?",
       beta=0.5,
   )
   model.fit(X_train, y_train)
   model.predict_proba(X_test)

A server on your own machine needs no key: with none set, no key header is sent. Pass ``api_key=...`` in the
``LLM`` if your server requires one.

Teaching the decision model your task
-------------------------------------

Used as it is, a decision model judges from the answers and the question alone. To let it learn your task,
fine-tune it on the same texts it will read: build them with ``answers_text`` from the fitted model's rules and
answers, so training and prediction see exactly the same format.

.. code-block:: python

   from think_reason_learn.rrf import answers_text

   answers = model.answers(X_train)  # probability of yes, one column per rule
   states = [answers_text(model.rules_, row) for row in answers.to_numpy()]
   # Train your decision model on (state, label) pairs, then serve it and refit
   # RRF with it as combiner_model. Kev's trainer takes one JSON line per sample:
   # {"state": state, "questions": {"positive": {"type": "noul",
   #   "instructions": combiner_question, "label": label == "successful"}}}

The threshold is tuned on the samples ``fit`` scores. If the decision model was trained on those same samples,
the threshold is optimistic: tune it on held-out samples instead.

Cost
----

The decision model is asked once per scored sample during ``fit`` and once per sample when predicting, on top of
the answerer's calls. Samples with identical answers are asked once, so ``estimate_cost`` counts these calls at
their most.
