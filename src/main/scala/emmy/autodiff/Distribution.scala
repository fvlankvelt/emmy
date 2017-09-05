package emmy.autodiff

import scalaz.Scalaz.Id

trait Distribution[U[_], V, S] {

  def sample(implicit model: Model): Sample[U, V, S]

  def observe(data: U[V]): Observation[U, V, S]
}

trait Stochast[V] {
  def logp(): Node[Id, V, Any]
}

trait Sample[U[_], V, S] extends Variable[U, V, S] with Stochast[V]

trait Observation[U[_], V, S] extends Node[U, V, S] with Stochast[V]
