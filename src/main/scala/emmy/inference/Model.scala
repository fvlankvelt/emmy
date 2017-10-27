package emmy.inference

import emmy.autodiff.{ContinuousVariable, EvaluationContext}
import emmy.distribution.Observation

trait Model {

  // new API - the Model contains a distribution over all variables
  // These distributions are updated in accordance with Bayes' Rule, when new evidence (observations) comes in

  def update[U[_], V, S](o: Seq[Observation[U, V, S]]): Model = this

  def sample(ec: EvaluationContext): ModelSample
}

trait ModelSample {

  def getSampleValue[U[_], S](n: ContinuousVariable[U, S]): U[Double]
}
