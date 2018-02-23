package emmy.inference.aevb

import emmy.autodiff._
import emmy.distribution.{ Categorical, Factor, Normal }

import scalaz.Scalaz.Id

sealed trait VariablePosterior {
  // original factor - the one that's used in local samples
  def O: Factor

  // current approximating distribution - the prior for next batch of observations
  def P: Factor

  // target distribution - approximates the posterior
  def Q: Factor

  def logQ: Expression[Id, Double, Any] = Q.logp

  def parameters: Seq[ParameterOptimizer]

  def next: VariablePosterior
}

case class ContinuousVariablePosterior[U[_], S](
    original:   Variable[U, Double, S],
    variable:   Variable[U, Double, S],
    muStart:    Option[Evaluable[U[Double]]] = None,
    sigmaStart: Option[Evaluable[U[Double]]] = None
)
  extends VariablePosterior {

  override val O: Variable[U, Double, S] = original

  override val P: Variable[U, Double, S] = variable

  implicit val fl = Floating.doubleFloating
  implicit val so = variable.so
  implicit val vOps: ContainerOps.Aux[U, S] = variable.ops

  val mu = new Parameter(muStart.getOrElse(variable.vt.map { vo ⇒ vo.zero }))
  val logSigma = new Parameter(sigmaStart.getOrElse(variable.vt.map { vo ⇒ vo.zero }))

  private val sigma = exp(logSigma)

  override val Q: Variable[U, Double, S] = Normal[U, S](mu, exp(logSigma)).sample

  override val logQ = Normal[U, S](mu.fix, exp(logSigma.fix)).factor(Q).logp

  override val parameters = Seq(
    ReparameterizedOptimizer(mu, sigma),
    ReparameterizedOptimizer(logSigma, Constant(mu.vt.map {
      vo ⇒ vo.div(vo.one, vo.sqrt(vo.fromInt(2)))
    }))
  //    ReparameterizedOptimizer(mu, Q, sigma.reciprocal(), sigma),
  //    ReparameterizedOptimizer(logSigma, Q, (variable - mu) / sigma, Constant(mu.vt.map {
  //      vo ⇒ vo.div(vo.one, vo.sqrt(vo.fromInt(2)))
  //    }))
  )

  override def next = {
    ContinuousVariablePosterior(O, Q, Some(mu.value), Some(logSigma.value))
  }
}

case class CategoricalVariablePosterior(
    original:    CategoricalVariable,
    variable:    CategoricalVariable,
    thetasStart: Option[Evaluable[IndexedSeq[Double]]] = None
)
  extends VariablePosterior {

  override val O: CategoricalVariable = original

  override val P: CategoricalVariable = variable

  private implicit val ops: ContainerOps.Aux[Id, Any] = variable.ops
  val thetas = new Parameter[IndexedSeq, Int](thetasStart.getOrElse(variable.K.map { k ⇒
    Array.fill(k)(0.0): IndexedSeq[Double]
  }))

  override val Q: CategoricalVariable = Categorical(exp(thetas)).sample

  override val parameters = Seq(ScoreFunctionOptimizer(thetas))

  override def next = {
    CategoricalVariablePosterior(O, Q, Some(thetas.value))
  }
}
