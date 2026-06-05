### Notes to be added to readme

What are the metrics used in evaluate()?

1. Faithfulness *REFERENCE FREE METRIC*
From ragas: The Faithfulness metric measures how factually consistent a response is with the retrieved context. It ranges from 0 to 1, with higher scores indicating better consistency.

A response is considered faithful if all its claims can be supported by the retrieved context.

To calculate this: 1. Identify all the claims in the response. 2. Check each claim to see if it can be inferred from the retrieved context. 3. Compute the faithfulness score using the formula:

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mtext>Faithfulness Score</mtext>
  <mo>=</mo>
  <mfrac>
    <mtext>Number of claims in the response supported by the retrieved context</mtext>
    <mtext>Total number of claims in the response</mtext>
  </mfrac>
</math>

A similar metric in ragas is Response Groundedness
requires: answer, contexts
Is the response supported by the contexts?
returns pass/fail 
but will not be used, since faithfulness tests this more aggressively. 

2. Answer Relevance *REFERENCE FREE METRIC*
The evaluation metric, Answer Relevancy, focuses on assessing how pertinent the generated answer is to the given prompt. A lower score is assigned to answers that are incomplete or contain redundant information and higher scores indicate better relevancy. This metric is computed using the question, the context and the answer.

The Answer Relevancy is defined as the mean cosine similarity of the original question to a number of artifical questions, which where generated (reverse engineered) based on the answer:
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mtext>answer relevancy</mtext>
  <mo>=</mo>
  <mfrac>
    <mn>1</mn>
    <mi>N</mi>
  </mfrac>
  <munderover>
    <mo data-mjx-texclass="OP">&#x2211;</mo>
    <mrow data-mjx-texclass="ORD">
      <mi>i</mi>
      <mo>=</mo>
      <mn>1</mn>
    </mrow>
    <mrow data-mjx-texclass="ORD">
      <mi>N</mi>
    </mrow>
  </munderover>
  <mi>c</mi>
  <mi>o</mi>
  <mi>s</mi>
  <mo stretchy="false">(</mo>
  <msub>
    <mi>E</mi>
    <mrow data-mjx-texclass="ORD">
      <msub>
        <mi>g</mi>
        <mi>i</mi>
      </msub>
    </mrow>
  </msub>
  <mo>,</mo>
  <msub>
    <mi>E</mi>
    <mi>o</mi>
  </msub>
  <mo stretchy="false">)</mo>
</math>

Where:

<math xmlns="http://www.w3.org/1998/Math/MathML">
  <msub>
    <mi>E</mi>
    <mrow data-mjx-texclass="ORD">
      <msub>
        <mi>g</mi>
        <mi>i</mi>
      </msub>
    </mrow>
  </msub>
</math>
is the embedding of the generated question 

<math xmlns="http://www.w3.org/1998/Math/MathML">
  <msub>
    <mi>E</mi>
    <mi>o</mi>
  </msub>
</math>
 is the embedding of the original question.

<math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>N</mi>
</math>
 is the number of generated questions, which is 3 default.

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mtext>Context Recall</mtext>
  <mo>=</mo>
  <mfrac>
    <mtext>Number of claims in the reference supported by the retrieved context</mtext>
    <mtext>Total number of claims in the reference</mtext>
  </mfrac>
</math>

3. Context Relevance 
requires: questions, contexts
Are the retrieved chunks relevant to the query?

4. Custom metric
Does the answer correctly represent the scope of the source (e.g., not saying "microplastics cause X in humans" when the paper only studied fish)?