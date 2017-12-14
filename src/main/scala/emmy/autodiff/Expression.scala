package emmy.autodiff

import emmy.autodiff.ContainerOps.Aux
import emmy.distribution.Observation
import emmy.inference.Sampler

import scalaz.Scalaz.Id

trait Visitor[R] {

  def visitParameter[U[_], S](o: Parameter[U, S]): R =
    visitNode(o)

  def visitSampler(o: Sampler): R =
    visitNode(o)

  def visitObservation[U[_], V, S](o: Observation[U, V, S]): R =
    visitNode(o)

  def visitContinuousVariable[U[_], S](v: ContinuousVariable[U, S]): R =
    visitNode(v)

  def visitCategoricalVariable(v: CategoricalVariable): R =
    visitNode(v)

  def visitNode(n: Node): R
}

trait Node {

  def visit[R](visitor: Visitor[R]): R = {
    visitor.visitNode(this)
  }

  def parents: Seq[Node] = Seq.empty

  override lazy val hashCode = super.hashCode()
}

trait Evaluable[+V] {
  self ⇒

  def apply(ec: SampleContext): V

  def map[W](fn: V ⇒ W): Evaluable[W] = new Evaluable[W] {

    override def apply(ec: SampleContext): W = {
      fn(self(ec))
    }

    override def toString() = {
      s"eval_map($self, $fn)"
    }
  }

}

object Evaluable {

  implicit def fromConstant[V](value: V): Evaluable[V] = new Evaluable[V] {

    override def apply(ec: SampleContext): V = value

    override def toString: String = {
      s"eval($value)"
    }
  }

  implicit def fromFn[V](fn: SampleContext ⇒ V): Evaluable[V] = new Evaluable[V] {

    override def apply(ec: SampleContext): V = fn(ec)

    override def toString: String = {
      s"eval($fn)"
    }
  }
}

case class SampleContext(seed: Int, iteration: Int)

trait GradientContext {

  def apply[U[_], V, S](n: Expression[U, V, S]): Evaluable[U[V]]

  def apply[W[_], U[_], V, T, S](
    n: Expression[U, V, S],
    v: Parameter[W, T]
  ): Gradient[W, U]
}

trait Expression[U[_], V, S] extends Node {

  type Shape = S

  implicit val ops: ContainerOps.Aux[U, Shape]

  implicit val so: ScalarOps[U[Double], U[V]]

  implicit def vt: Evaluable[ValueOps[U, V, S]]

  def eval(ec: GradientContext): Evaluable[U[V]]

  def grad[W[_], T](
    gc: GradientContext,
    v:  Parameter[W, T]
  ): Gradient[W, U] = None

  def unary_-(): Expression[U, V, S] =
    UnaryExpression[U, V, S](this, new EvaluableValueFunc[V] {
      override def name: String = "neg"

      override def grad(gc: SampleContext, v: V) = {
        val valueVT = vt(gc).valueVT
        valueVT.negate(valueVT.one)
      }

      override def apply(ec: SampleContext, v: V) = {
        val vvt = vt(ec).valueVT
        vvt.negate(v)
      }
    })

  lazy val toDouble: Expression[U, Double, S] = {
    val self = this
    new Expression[U, Double, S] {

      override val parents = Seq(self)

      override implicit val ops: Aux[U, Shape] = self.ops

      override implicit def vt: Evaluable[ValueOps[U, Double, S]] =
        self.vt.map { up ⇒
          ValueOps(Floating.doubleFloating, ops, up.shape)
        }

      override implicit val so = ScalarOps.liftBoth[U, Double, Double](ScalarOps.doubleOps, ops)

      override def eval(ec: GradientContext): Evaluable[U[Double]] = {
        val up = self.eval(ec)
        ctx ⇒ {
          val valueOps = self.vt(ctx)
          val upVal = up(ctx)
          val valT = valueOps.valueVT
          ops.map(upVal)(valT.toDouble)
        }
      }

      override def grad[W[_], T](gc: GradientContext, v: Parameter[W, T]) = {
        self.grad(gc, v)
      }

      override def toString = s"double($self)"
    }
  }

  def reciprocal(): Expression[U, V, S] =
    Reciprocal(this)

  // element-wise ops

  def *(rhs: Expression[U, V, S]): Expression[U, V, S] =
    Multiply(this, rhs)

  def /(rhs: Expression[U, V, S]): Expression[U, V, S] =
    Multiply(this, rhs.reciprocal())

  def +(rhs: Expression[U, V, S]): Expression[U, V, S] =
    Add(this, rhs)

  def -(rhs: Expression[U, V, S]): Expression[U, V, S] =
    Add(this, -rhs)

  // scalar ops

  def *[W](value: W)(implicit sOps: ScalarOps[V, W]): Expression[U, V, S] =
    UnaryExpression[U, V, S](this, new EvaluableValueFunc[V] {
      val name = s"${value} *"

      override def grad(gc: SampleContext, v: V) =
        sOps.times(vt(gc).valueVT.one, value)

      override def apply(ec: SampleContext, v: V) =
        sOps.times(v, value)
    })

  def /[W](value: W)(implicit sOps: ScalarOps[V, W]): Expression[U, V, S] =
    UnaryExpression[U, V, S](this, new EvaluableValueFunc[V] {
      val name = s"inv(${value})*"

      override def grad(gc: SampleContext, v: V) =
        sOps.div(vt(gc).valueVT.one, value)

      override def apply(ec: SampleContext, v: V) =
        sOps.div(v, value)
    })

  def +[W](rhs: W)(implicit sOps: ScalarOps[V, W]): Expression[U, V, S] = {
    UnaryExpression[U, V, S](this, new EvaluableValueFunc[V] {
      val name = s"${rhs}+"

      override def grad(gc: SampleContext, v: V) =
        vt(gc).valueVT.one

      override def apply(ec: SampleContext, v: V) =
        sOps.plus(v, rhs)
    })
  }

  def -[W](rhs: W)(implicit sOps: ScalarOps[V, W]): Expression[U, V, S] = {
    UnaryExpression[U, V, S](this, new EvaluableValueFunc[V] {
      val name = s"-${rhs}+"

      override def grad(gc: SampleContext, v: V) =
        vt(gc).valueVT.one

      override def apply(ec: SampleContext, v: V) =
        sOps.minus(v, rhs)
    })
  }
}
