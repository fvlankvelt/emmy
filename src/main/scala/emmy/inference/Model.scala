package emmy.inference

import emmy.autodiff.{ SampleContext, Variable }
import emmy.distribution.Factor

trait Model {

  // new API - the Model contains a distribution over all variables
  // These distributions are updated in accordance with Bayes' Rule, when new evidence (observations) comes in

  def update(o: Seq[Factor]): Model = this

  def sample[U[_], V, S](v: Variable[U, V, S], ec: SampleContext): U[V]
}
