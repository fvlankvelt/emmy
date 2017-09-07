package emmy.distribution

import emmy.autodiff.{ConstantLike, Node, Variable}

import scalaz.Scalaz.Id

trait Distribution[U[_], V, S] {

  def sample: Variable[U, V, S]

  def observe(data: U[V]): Observation[U, V, S]
}

trait ValueDistribution[U[_], V, S] {

  def sample: U[V]
}

trait Stochast[V] {
  def logp(): Node[Id, V, Any]
}

trait Observation[U[_], V, S] extends ConstantLike[U, V, S] with Stochast[V]
