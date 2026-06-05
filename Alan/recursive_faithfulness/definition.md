 # Recursive Faithfulness
$F(x):=RAGAS DEFINITION$ // counts number of correct citations divided by number of citations
$F_R(x):=CorrectCitations \cdot Faithfulness of Cited Documents$ // counts faithfulness of correct citations

TODO: add in a decay factor

Reduction:
1/N Sum
Sum

Other options:
Incorrect citation of a faithful document is weighted more than unfaithful reference to unfaithful? Eh. But, where denominator is sum of faithfulness of all cited works, ... I'm not sure if it's meaningful.

TODO: Examine metric performance with Citation Network standin (number of citations of document == ~importance)

Later:
Integrate a regex/LLM->Jinja processing whereby objective grading of faithfulness can occur where items of document A are replaced with items from document B and claim of document B is checked verbatim on modified document A

Further: This can be connected to Ergodic Hierarchy, where `phase flow` is the sequence of states terminating from an initial state, $\phi(t)$, asking whether the implications of A and B are not contradictory, and whether the implications of A and B (with rewriting) are identical.
