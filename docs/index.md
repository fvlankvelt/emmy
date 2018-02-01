# Emmy
Probabilistic programming language embedded in Scala.  It's focus is on
scalability, with inference algorithms like variational bayes and the No-U-Turn
sampler.

## Model specification
The language allows the user to specify a model of data generation.
Observations can be bound to the (hierarchical) distributions so its parameters
can be inferred.

For scalability, it is imperative to be able to calculate the gradient of the
(log) posterior probability.  To do this, [automatic
differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) has
been implemented.  A DAG is constructed of operators, variables, constants and
distributions.  The gradient is evaluated at the assigned value.  (this in
contrast to symbolic differentiation which would be much harder to implement)

Value types are flexible to the extent that new "container" types can be easily
added.  The only restriction is that a container type constructor U has one
type argument.  The intention is to be able to adapt to the shape of the data
rather than limiting to a predefined format.

Formulas can be naturally expressed:
```scala
val logp = sum(alpha * log(beta) + (alpha - 1.0) * log(z) - beta * z - lgamma(alpha))
```
where `alpha`, `beta` and `z` are `Node`s which evaluate to either a Double or
a container of doubles.  Operators (`*`, `+`, `-`) and functions (`log`,
`lgamma`) operate element-wise, with `sum` reducing the container to a single
value.  The result `logp` is also a `Node`, of which the autodiff algorithm can
take the gradient.

A more elaborate example for linear regression:
```scala
// specify variables with priors
val a = Normal(0.0, 1.0).sample
val b = Normal(List(0.0, 0.0), List(1.0, 1.0)).sample
val e = Normal(1.0, 1.0).sample

// the data in (X, Y) tuples
val data = List(
  (List(1.0, 2.0), 0.5),
  (List(2.0, 1.0), 1.0)
)

// bind the data to the linear model
val observations = data.map {
  case (x, y) =>
    val s = a + sum(x * b)
    Normal(s, e).observe(y)
}
```
which specifies an intercept `a`, a slope `b` and noise `e`.  Each of these variables are specified as samples from a prior.

## Variational Inference
Keeping with the spirit of Bayes' formula, it is possible to update the distributions of model parameters by feeding data to the model.  This is implemented using Auto-Encoding Variational Bayes, which uses the generative model as the decoder and the mean-field approximation the encoder.

For efficiency, it is possible to feed the model a minibatch of observations.  An isomorphic model is constructed when this happens, so a stream of data can be processed - it is not necessary to have all data up-front.

In the following example, we'll generate a set of data using the same model that is fitted - demonstrating the power of VI to recover the parameters.
```scala
// data generation - demo/test
val data = {
  val alpha = 1.0
  val sigma = 1.0
  val beta = List(1.0, 2.5)

  (for {_ <- 0 until 100} yield {
    val X = List(Random.nextGaussian(), 0.2 * Random.nextGaussian())
    val Y = alpha + X(0) * beta(0) + X(1) * beta(1) + Random.nextGaussian() * sigma
    (X, Y)
  }).toList
}

// model parameters with their priors
val a = Normal(0.0, 1.0).sample
val b = Normal(List(0.0, 0.0), List(1.0, 1.0)).sample
val e = Normal(0.0, 1.0).sample
val model = AEVBModel[Double](Seq[Node](a, b, e))

// infer parameter values from observations
val observations = data.map {
  case (x, y) =>
    val s = a + sum(x * b)
    Normal(s, e).observe(y)
}
val newModel = model.update(observations)
val aDist = newModel.distributionOf(a)
val bDist = newModel.distributionOf(b)
val eDist = newModel.distributionOf(e)
println(s"a: mu = ${aDist._1}, sigma = ${aDist._2}")
println(s"b: mu = ${bDist._1}, sigma = ${bDist._2}")
println(s"e: mu = ${eDist._1}, sigma = ${eDist._2}")
```
will output the following variational approximation:
```
a: mu = 1.251511998474368, sigma = 0.1077319388946663
b: mu = List(0.9111702925643559, 2.271832642370476), sigma = List(0.12496776400666693, 0.5165769822733974)
e: mu = 1.0930019599344285, sigma = 0.083023021331367
```
