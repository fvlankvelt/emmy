package emmy.inference

import emmy.autodiff.{ContainerOps, ValueOps, Variable}
import emmy.distribution.Observation


trait Model {

  // new API - the Model contains a distribution over all variables
  // These distributions are updated in accordance with Bayes' Rule, when new evidence (observations) comes in

  def update[U[_], V, S](o: Observation[U, V, S]): Model

  def sample(): ModelSample
}

trait ModelSample {

  def getSampleValue[U[_], V, S](n: Variable[U, V, S])(implicit vo: ValueOps[U, V, S], ops: ContainerOps.Aux[U, S]): U[V]
}
