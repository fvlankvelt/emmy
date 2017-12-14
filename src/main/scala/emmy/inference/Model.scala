package emmy.inference

import emmy.autodiff.{ SampleContext, Variable }
import emmy.distribution.Observation

trait Model {

  // new API - the Model contains a distribution over all variables
  // These distributions are updated in accordance with Bayes' Rule, when new evidence (observations) comes in

  def update[U[_], V, S](o: Seq[Observation[U, V, S]]): Model = this

  def sample[U[_], V, S](v: Variable[U, V, S], ec: SampleContext): U[V]
}
