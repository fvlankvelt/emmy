package emmy.distribution

import emmy.autodiff.ContainerOps.Aux
import emmy.autodiff._

import scalaz.Scalaz
import scalaz.Scalaz.Id

trait SelectFactor[U[_], V, S] extends Factor with Node {

  def index: Expression[Id, Int, Any]

  def logs: Seq[Expression[Id, Double, Any]]

  override val logp: Expression[Scalaz.Id, Double, Any] = {
    val self = this
    new Expression[Id, Double, Any] {

      override def parents = (logs: Seq[Node]) :+ self

      override implicit val ops: Aux[Scalaz.Id, Shape] =
        ContainerOps.idOps

      override implicit val so: ScalarOps[Scalaz.Id[Double], Scalaz.Id[Double]] =
        ScalarOps.doubleOps

      override implicit def vt: Evaluable[ValueOps[Scalaz.Id, Double, Any]] =
        ValueOps(Floating.doubleFloating, ops, null)

      override def eval(ec: GradientContext): Evaluable[Scalaz.Id[Double]] = {
        val cLogs = logs.map(ec(_))
        val cIndex = ec(index)
        ctx => {
          val idx = cIndex(ctx)
          cLogs(idx)(ctx)
        }
      }

      override def grad[W[_], T](gc: GradientContext,
                                 v: Parameter[W, T]) = {
        val cGrads = logs.map(log => gc(log, v))
        if (cGrads.forall(_.isEmpty))
          None
        else {
          val cIndex = gc(index)
          Some { ctx =>
            val idx = cIndex(ctx)
            val grad = cGrads(idx)
            val vOps = v.vt(ctx)
            grad.map { g => g(ctx) }.getOrElse(vOps.zero)
          }
        }
      }

      override def toString = s"logp($self)"
    }
  }

}

trait Select[U[_], V, S] extends Distribution[U, V, S] {

  def multi: Categorical

  def clusters: Seq[Distribution[U, V, S]]

  implicit val distOps: ContainerOps.Aux[U, S]
  implicit val distScalar: ScalarOps[U[Double], U[V]]

  trait SelectVariable extends Expression[U, V, S] with SelectFactor[U, V, S] {

    def observations: Seq[Variable[U, V, S]]

    override val logs = observations.map(_.logp)

    override val parents: Seq[Node] =
      observations.flatMap(_.parents) :+ index

    override implicit val ops: ContainerOps.Aux[U, S] = distOps

    override implicit val so: ScalarOps[U[Double], U[V]] = distScalar

    override implicit def vt: Evaluable[ValueOps[U, V, S]] = observations.head.vt

    override def eval(ec: GradientContext) = {
      val cObservations = observations.map { ec(_) }
      val cIndex = ec(index)
      ctx => {
        val idx = cIndex(ctx)
        cObservations(idx)(ctx)
      }
    }

    override def grad[W[_], T](gc: GradientContext, v: Parameter[W, T]) = {
      val wOps = v.ops
      val cGrads = observations.map {
        gc(_, v)
      }
      val cIndex = gc(index)
      if (cGrads.forall(_.isEmpty)) {
        None
      } else {
        Some { ctx =>
          val idx = cIndex(ctx)
          val grad = cGrads(idx)
          val valT = vt(ctx).forDouble
          val eVt = v.vt(ctx)
          grad.map { g =>
            g(ctx)
          }.getOrElse {
            wOps.fill(eVt.shape, valT.zero)
          }
        }
      }
    }
  }

  override def observe(data: U[V]): Observation[U, V, S] = {
    val sIndex = multi.sample
    val observations = clusters.map(_.observe(data))

    new Observation[U, V, S] with SelectFactor[U, V, S] {

      override val index = sIndex

      override val logs = observations.map(_.logp)

      override val parents: Seq[Node] =
        observations.flatMap(_.parents) :+ index

      override implicit val ops: ContainerOps.Aux[U, S] = distOps

      override implicit val so: ScalarOps[U[Double], U[V]] = distScalar

      override implicit val vt = observations.head.vt

      override def value: Evaluable[U[V]] =
        data

      override def toString = s"select($multi, $data)"
    }
  }
}

object Select {

  def apply[U[_], S](argMulti: Categorical, argClusters: Seq[Distribution[U, Double, S]])(implicit
    aOps: ContainerOps.Aux[U, S],
                                                                                          sOps: ScalarOps[U[Double], U[Double]]
  ): Distribution[U, Double, S] =
    new Select[U, Double, S] {

      val multi: Categorical = argMulti
      val clusters: Seq[Distribution[U, Double, S]] = argClusters

      override implicit val distOps: Aux[U, S] = aOps
      override implicit val distScalar: ScalarOps[U[Double], U[Double]] = sOps

      override def sample: Variable[U, Double, S] = {
        new SelectVariable with ContinuousVariable[U, S] {

          override val index: CategoricalSample = multi.sample

          override val observations: Seq[Variable[U, Double, S]] = clusters.map(_.sample)
        }
      }
    }

  def apply(argMulti: Categorical, argClusters: Seq[Categorical]): Distribution[Id, Int, Any] =
    new Select[Id, Int, Any] {

      val multi: Categorical = argMulti
      val clusters: Seq[Categorical] = argClusters

      override implicit val distOps: Aux[Id, Any] = ContainerOps.idOps
      override implicit val distScalar: ScalarOps[Double, Int] = ScalarOps.intDoubleOps

      override def sample: Variable[Id, Int, Any] = {
        new SelectVariable with CategoricalVariable {

          override val index: CategoricalSample = multi.sample

          override val observations: Seq[Variable[Id, Int, Any]] = clusters.map(_.sample)

          override val K = clusters.head.thetas.vt.map(_.shape)
        }
      }
    }

}
