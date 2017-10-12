package emmy.distribution

import emmy.autodiff.{ConstantLike, Expression, Variable}

import scalaz.Scalaz.Id

trait Distribution[U[_], V, S] {

  def sample: Variable[U, S]

  def observe(data: U[V]): Observation[U, V, S]
}

trait ValueDistribution[U[_], V, S] {

  def sample: U[V]
}

trait Stochast {
  def logp(): Expression[Id, Double, Any]
}

trait Observation[U[_], V, S] extends ConstantLike[U, V, S] with Stochast
