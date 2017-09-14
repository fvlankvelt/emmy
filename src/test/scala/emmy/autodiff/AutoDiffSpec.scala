package emmy.autodiff

import breeze.numerics.abs
import emmy.autodiff.ContainerOps.Aux
import emmy.distribution.{Normal, Observation}
import emmy.inference.{Model, ModelSample}
import org.scalatest.FlatSpec

import scala.annotation.tailrec
import scala.collection.mutable
import scala.util.Random
import scalaz.Scalaz._

class AutoDiffSpec extends FlatSpec {

  val gc = new GradientContext[Double] {

    private val cache = mutable.HashMap[AnyRef, Any]()

    override def apply[U[_], S](n: Expression[U, Double, S]): U[Double] = {
      n match {
        case v: TestVariable[U, Double, S] => v.value
        case v: Variable[U, Double, S] => cache.getOrElseUpdate(v, v.vt.rnd).asInstanceOf[U[Double]]
        case _ => n.apply(this)
      }
    }

    override def apply[W[_], U[_], T, S](n: Expression[U, Double, S], v: Variable[W, Double, T])(implicit wOps: Aux[W, T]): W[U[Double]] = {
      n.grad(this, v)
    }
  }

  val ec: EvaluationContext[Double] = gc

  case class TestVariable[U[_], V, S](value: U[V])
                                     (implicit
                                      val vo: ValueOps[U, V, S],
                                      val ops: ContainerOps.Aux[U, S])
    extends Variable[U, V, S] {

    override def shape = ops.shapeOf(value)

    override def apply(ec: EvaluationContext[V]) = value

    override def logp() = ???

    override implicit val vt = vo.bind(ops.shapeOf(value))
  }

  "AD" should "calculate scalar derivative" in {
    val x = TestVariable[Id, Double, Any](2.0)
    val y = x * x
    assert(y(ec) == 4.0)

    val z: Double = y.grad(gc, x)
    assert(z == 4.0)
  }

  it should "calculate vector derivative on List" in {
    val x = TestVariable[List, Double, Int](List(1.0, 2.0))
    val y = x * x
    assert(y(ec) == List(1.0, 4.0))

    val z = y.grad(gc, x)
    assert(z == List(List(2.0, 0.0), List(0.0, 4.0)))
  }

  it should "calculate derivative of a scalar function" in {
    val x = TestVariable[Id, Double, Any](2.0)
    val y = log(x)
    assert(y(ec) == scala.math.log(2.0))

    val z: Double = y.grad(gc, x)
    assert(z == 0.5)
  }

  it should "calculate derivative of a function applied to a list" in {
    val x = TestVariable[List, Double, Int](List(1.0, 2.0))
    val y = log(x)
    assert(y(ec) == List(0.0, scala.math.log(2.0)))

    val z = y.grad(gc, x)
    assert(z == List(List(1.0, 0.0), List(0.0, 0.5)))
  }

  it should "calculate probability of observation" in {
    val mu = TestVariable[List, Double, Int](List(0.0, 0.0))
    val sigma = TestVariable[List, Double, Int](List(1.0, 1.0))

    val normal = Normal(mu, sigma)
    val observation = normal.observe(List(1.0, 2.0))
    println(observation.logp()(ec))
  }

  it should "be able to implement linear regression" in {
    val a = Normal(0.0, 1.0).sample
    val b = Normal(List(0.0, 0.0), List(1.0, 1.0)).sample
    val e = Normal(1.0, 1.0).sample

    val data = List(
      (List(1.0, 2.0), 0.5),
      (List(2.0, 1.0), 1.0)
    )

    val observations = data.map {
      case (x, y) =>
        val s = a + sum(x * b)
        Normal(s, e).observe(y)
    }
    val logp = observations.map(_.logp()).sum +
      a.logp() + b.logp() + e.logp()
    println(logp(ec))

    val g_a: Double = logp.grad(gc, a)
    println(g_a)
  }

  it should "update variational parameters for each (minibatch of) data point(s)" in {
    val mu = Normal(0.0, 1.0).sample
    val sigma = Normal(0.0, 0.5).sample

    val initialModel = SimpleModel[Double](Seq(mu, sigma))
    val dist = Normal(mu, exp(sigma))

    var model = initialModel
    while(true) {
      val data = for {_ <- 0 until 100} yield {
        0.3 + Random.nextGaussian() * scala.math.exp(0.2)
      }

      val observations = data.map { d => dist.observe(d) }
      val newModel = model.update(observations)
      print(newModel)
      model = newModel
    }
  }

  case class SimpleModel[V] private[SimpleModel](
                                                  nodes: Set[Node],
                                                  globalVars: Map[Node, Any]
                                                )(implicit fl: Floating[V])
    extends Model[V] {

    import SimpleModel._

    override def update[U[_], S](observations: Seq[Observation[U, V, S]]) = {

      // find new nodes, new variables & their (log) probability
      val (_, samplerBuilders, logp) = collectVars(
        nodes,
        Set.empty[SamplerBuilder[W forSome {type W[_]}, V, _]],
        fl.zero,
        observations
      )

      // initialize new variables by sampling their prior
      // (based on current distributions for already known variables)
      val localVars = if (samplerBuilders.nonEmpty) {
        initialize(samplerBuilders, sample)
      } else {
        Map.empty
      }

      // logP is the sum of
      // - the likelihood of observation log(p(x|\theta)), and
      // - the prior log(p(\theta)) of the local variables
      //
      // Add the log prior of global variables to get the full
      // objective function to optimize
      val totalLogP = globalVars.values.foldLeft(logp) { case (curLogp, variable) =>
        curLogp + variable.asInstanceOf[Sampler[({type U[_]})#U, V, _]].variable.logp()
      }

      // update variables by taking observations into account
      val samplers = (globalVars ++ localVars).map {
        _._2.asInstanceOf[Sampler[({type U[_]})#U, V, _]]
      }

      @tailrec
      def iterate(iter: Int, samplers: Iterable[Sampler[({type U[_]})#U, V, _]]): Iterable[Sampler[({type U[_]})#U, V, _]] = {
        val variables = samplers.map { s => (s.variable: Node) -> (s: Any) }.toMap
        val rho = fl.div(fl.one, fl.fromInt(iter + 1000))
        val modelSample = new ModelSample[V] {
          override def getSampleValue[U[_], S](n: Variable[U, V, S]): U[V] =
            variables(n).asInstanceOf[Sampler[U, V, S]].sample()
        }
        val gc = new ModelGradientContext[V](modelSample)
        val updatedWithDelta = samplers.map { anyS =>
          val sampler = anyS.asInstanceOf[Sampler[({type U[_]})#U, V, _]]
          sampler.update(totalLogP, gc, rho)
        }.toMap[Sampler[({type U[_]})#U, V, _], V]

        val totalDelta = updatedWithDelta.values.sum
        if (fl.lt(totalDelta, fl.div(fl.one, fl.fromInt(1000)))) {
          updatedWithDelta.keys
        } else {
          iterate(iter + 1, updatedWithDelta.keys)
        }
      }

      val newSamplers = iterate(0, samplers)

      SimpleModel(nodes,
        newSamplers.filter { sampler =>
          globalVars.contains(sampler.variable)
        }.map { sampler =>
          (sampler.variable: Node) -> sampler
        }.toMap
      )
    }

    override def sample() = new ModelSample[V] {
      override def getSampleValue[U[_], S](n: Variable[U, V, S]): U[V] =
        globalVars(n).asInstanceOf[Sampler[U, V, S]].sample()
    }

  }

  object SimpleModel {

    private[SimpleModel] def initialize[V]
    (
      builders: Set[SamplerBuilder[W forSome {type W[_]}, V, _]],
      prior: () => ModelSample[V]
    ): Map[Node, Any] = {
      for {_ <- 0 until 100} {
        val modelSample = prior()
        val newVariables = builders.map {
          _.variable: Node
        }
        val ec = new ModelEvaluationContext[V](modelSample, newVariables)
        for {initializer <- builders} {
          initializer.eval(ec)
        }
      }
      builders.toSeq.map { b =>
        b.variable -> b.build()
      }.toMap
    }

    private[SimpleModel] def collectVars[V]
    (
      visited: Set[Node],
      vars: Set[SamplerBuilder[W forSome {type W[_]}, V, _]],
      lp: Expression[Id, V, Any],
      nodes: Seq[Node]
    ): (Set[Node], Set[SamplerBuilder[W forSome {type W[_]}, V, _]], Expression[Id, V, Any]) = {
      nodes.foldLeft((visited, vars, lp)) {
        case ((curvis, curvars, curlogp), p) =>
          p match {
            case _ if curvis.contains(p) =>
              (curvis, curvars, curlogp)
            case o: Observation[W forSome {type W[_]}, V, _] =>
              collectVars(curvis + p, curvars, curlogp + o.logp(), p.parents)
            case v: Variable[W forSome {type W[_]}, V, _] =>
              collectVars(curvis + p, curvars + SamplerBuilder(v), curlogp + v.logp(), p.parents)
            case _ =>
              collectVars(curvis + p, curvars, curlogp, p.parents)
          }
      }
    }

    def apply[V](global: Seq[Node])(implicit fl: Floating[V]): SimpleModel[V] = {

      // find new nodes, new variables & their (log) probability
      val (_, builders, logp) = collectVars(
        Set.empty,
        Set.empty[SamplerBuilder[W forSome {type W[_]}, V, _]],
        fl.zero,
        global
      )

      val globalSamplers = initialize(builders, () => {
        new ModelSample[V] {
          override def getSampleValue[U[_], S](n: Variable[U, V, S]): U[V] =
            throw new UnsupportedOperationException("Global priors cannot be initialized with dependencies on variables")
        }
      })

      SimpleModel[V](global.toSet, globalSamplers)
    }
  }

  class ModelEvaluationContext[V](modelSample: ModelSample[V], newVariables: Set[Node]) extends EvaluationContext[V] {

    private val cache = mutable.HashMap[AnyRef, Any]()

    override def apply[U[_], S](n: Expression[U, V, S]): U[V] =
      n match {
        case v: Variable[U, V, S] if !newVariables.contains(v) =>
          cache.getOrElseUpdate(n, modelSample.getSampleValue(v))
            .asInstanceOf[U[V]]
        case _ =>
          cache.getOrElseUpdate(n, n.apply(this))
            .asInstanceOf[U[V]]
      }

  }

  class ModelGradientContext[V](modelSample: ModelSample[V]) extends GradientContext[V] {

    private val cache = mutable.HashMap[AnyRef, Any]()

    override def apply[U[_], S](n: Expression[U, V, S]): U[V] =
      n match {
        case v: Variable[U, V, S] =>
          cache.getOrElseUpdate(n, modelSample.getSampleValue(v))
            .asInstanceOf[U[V]]
        case _ =>
          cache.getOrElseUpdate(n, n.apply(this))
            .asInstanceOf[U[V]]
      }

    override def apply[W[_], U[_], T, S](n: Expression[U, V, S], v: Variable[W, V, T])(implicit wOps: Aux[W, T]): W[U[V]] = {
      n.grad(this, v)
    }
  }

  case class SamplerBuilder[U[_], V, S](variable: Variable[U, V, S]) {
    private val samples: mutable.Buffer[U[V]] = mutable.Buffer.empty

    def eval(ec: EvaluationContext[V]): Unit = {
      samples += ec(variable)
    }

    def build(): Sampler[U, V, S] = {
      implicit val numUV = variable.vt
      val size = samples.length
      val mu: U[V] = numUV.div(samples.sum(numUV), numUV.fromInt(size))
      val sigma2 = samples.map { x =>
        val delta = numUV.minus(x, mu)
        numUV.times(delta, delta)
      }.sum(numUV)
      val ratio = numUV.div(sigma2, numUV.fromInt(size - 1))
      val sigma = numUV.sqrt(ratio)
      new Sampler(variable, mu, sigma)
    }
  }

  class Sampler[U[_], V, S](val variable: Variable[U, V, S], val mu: U[V], sigma: U[V]) {

    /*
    println(s"new sampler for ${variable}: ${mu}, ${sigma}")
    if (mu.asInstanceOf[Double].isNaN ||
      abs(mu.asInstanceOf[Double]) > 20.0 ||
      abs(sigma.asInstanceOf[Double]) > 20.0 ||
      sigma.asInstanceOf[Double].isNaN) {
      assert(false)
    }
    */

    def gradValue(value: U[V]): U[V] = {
      val vt = variable.vt
      val delta = vt.minus(mu, value)
      vt.div(delta, vt.times(sigma, sigma))
    }

    /**
      * Update (\mu, \sigma) by taking a natural gradient step of size \rho.
      * The value is decomposed as \value = \mu + \epsilon * \sigma.
      * Updates are
      *
      * \mu' = (1 - \rho) * \mu    +
      * \rho * \sigma**2 * (\gradP - \gradQ)
      *
      * \sigma' = (1 - \rho) * \sigma +
      * \rho * (\sigma**2 / 2) * \epsilon * (\gradP - \gradQ)
      *
      * The factors \sigma**2 and \sigma**2/2, respectively, are due to
      * the conversion to natural gradient.  The \epsilon factor is
      * the jacobian d\theta/d\sigma.  (similar factor for \mu is 1)
      */
    def update(logP: Expression[Id, V, Any], gc: GradientContext[V], rho: V): (Sampler[U, V, S], V) = {
      val vt = variable.vt
      val fl = vt.valueVT
      val value = gc(variable)
      implicit val ops = variable.ops
      val gradP = gc(logP, variable)
      val gradQ = gradValue(value)
      val gradDelta = variable.vt.minus(gradQ, gradP)

      val newMu = vt.plus(
        vt.times(
          variable.ops.fill(variable.shape, fl.minus(fl.one, rho)),
          mu
        ),
        vt.times(
          vt.times(sigma, sigma),
          vt.times(variable.ops.fill(variable.shape, rho), gradDelta)
        )
      )

      val lambda = vt.log(sigma)
      val newLambda =
        vt.plus(
          vt.times(
            lambda, variable.ops.fill(variable.shape, fl.minus(fl.one, rho))
          ),
          vt.times(
            vt.div(vt.minus(value, mu), vt.fromInt(2)),
            vt.times(variable.ops.fill(variable.shape, rho), gradDelta)
          )
        )
      val newSigma = vt.exp(vt.div(vt.plus(newLambda, lambda), vt.fromInt(2)))
      val newSampler = new Sampler[U, V, S](variable, newMu, newSigma)
      (newSampler, delta(newSampler))
    }

    def delta(other: Sampler[U, V, S]): V = {
      implicit val vt = variable.vt
      implicit val fl = vt.valueVT
      val ops = variable.ops
      ops.foldLeft(vt.abs(vt.div(vt.minus(mu, other.mu), sigma)))(fl.zero)(fl.sum)
    }

    def gradMu(value: U[V]): U[V] = {
      val vt = variable.vt
      val delta = vt.minus(value, mu)
      vt.div(delta, vt.times(sigma, sigma))
    }

    def gradSigma(value: U[V]): U[V] = {
      val vt = variable.vt
      val delta = vt.minus(value, mu)
      vt.minus(
        vt.div(
          vt.times(delta, delta),
          vt.times(vt.times(sigma, sigma), sigma)
        ),
        vt.div(vt.one, sigma)
      )
    }

    def sample(): U[V] = {
      val vt = variable.vt
      val value = vt.plus(mu, vt.times(vt.rnd, sigma))
//      println(s"sampling ${variable}: ${mu}, ${sigma} => $value")
      value
    }
  }

}
