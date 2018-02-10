# Emmy
Probabilistic programming language embedded in Scala.  It's focus is on
scalability, with inference algorithms like variational bayes and the No-U-Turn
sampler.

# Features
Emmy provides a language for (bayesian) inference and learning on the JVM.  
When constructing a model of the data, i.e. a generative process based on
some parameters, an abstract syntax tree (AST) of *Expressions* is constructed.

#### Flexible data types
Data container types include various *scala collections*, to make their use
as natural as possible within a scala machine learning pipeline.  Values can
be integer- (`Int`) or real-valued (`Double`).  Distributions can be
continuous (`Normal`, `Gamma`) or discrete (`Categorical`).

#### Automatic Differentiation
When optimizing approximations to the posterior distribution, automatic
differentiation is used.  This takes a large, error-prone, burden off the process
of implementing a new model.

#### Compiling to a function
Since values and derivatives are calculated many times during Monte Carlo 
evaluations, these are compiled to a function.  Memoization reduces multiple
uses of the same value to a single evaluation.

#### Dynamic Graph Construction
To allow for maximum flexibility for inference, graphs can be dynamic - they
are reconstructed for each mini-batch of data.

# Documentation
For documentation and examples, see <http://vanlankvelt.com/emmy>.
