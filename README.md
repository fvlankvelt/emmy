# Emmy
Probabilistic programming language embedded in Scala.  It's focus is on
scalability, with inference algorithms like variational bayes and the No-U-Turn
sampler.

## AutoDiff
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
val a = Normal(
  Constant[Id, Double, Any](0.0),
  Constant[Id, Double, Any](1.0)
).sample

val b = Normal(
  Constant(List(0.0, 0.0)),
  Constant(List(1.0, 1.0))
).sample

val e = Normal(
  Constant[Id, Double, Any](1.0),
  Constant[Id, Double, Any](1.0)
).sample

// the data in (X, Y) tuples
val data = List(
  (List(1.0, 2.0), 0.5),
  (List(2.0, 1.0), 1.0)
)

// bind the data to the linear model
val observations = data.map {
  case (x, y) =>
    val cst = Constant[List, Double, Int](x)
    val s = a + sum(cst * b)
    Normal(s, e).observe(y)
}

// calculate log posterior
val logp = observations.map(_.logp()).sum +
  a.logp() + b.logp() + e.logp()
println(logp())

val g_a: Double = logp.grad(a)
println(g_a)
```
which specifies an intercept `a`, a slope `b` and noise `e`.  Each of these variables are specified as samples from a prior.
