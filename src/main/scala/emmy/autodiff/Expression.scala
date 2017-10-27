package emmy.autodiff
import emmy.autodiff.ContainerOps.Aux
import emmy.distribution.Observation

trait Visitor[R] {

  def visitObservation[U[_], V, S](o: Observation[U, V, S]): R

  def visitVariable[U[_], V, S](v: ContinuousVariable[U, S]): R

  def visitNode(n: Node): R
}

trait Node {

  def visit[R](visitor: Visitor[R]): R

  def parents: Seq[Node] = Seq.empty
}

trait Evaluable[+V] {
  self ⇒

  def apply(ec: EvaluationContext): V

  def map[W](fn: V ⇒ W): Evaluable[W] = new Evaluable[W] {

    override def apply(ec: EvaluationContext): W = {
      fn(self(ec))
    }

    override def toString() = {
      s"eval_map($self, $fn)"
    }
  }

}

object Evaluable {

  implicit def fromConstant[V](value: V): Evaluable[V] = new Evaluable[V] {

    override def apply(ec: EvaluationContext): V = value

    override def toString() = {
      s"eval($value)"
    }
  }
}

trait EvaluationContext {

  def apply[U[_], V, S](n: Expression[U, V, S]): U[V]
}

trait GradientContext extends EvaluationContext {

  def apply[W[_], U[_], V, T, S](n: Expression[U, V, S], v: ContinuousVariable[W, T])(implicit wOps: ContainerOps.Aux[W, T]): W[U[Double]]
}

trait Expression[U[_], V, S] extends Node with Evaluable[U[V]] {

  type Shape = S

  implicit val ops: ContainerOps.Aux[U, Shape]

  implicit val so: ScalarOps[U[Double], U[V]]

  implicit def vt: Evaluable[ValueOps[U, V, S]]

  def visit[R](visitor: Visitor[R]) = {
    visitor.visitNode(this)
  }

  def apply(ec: EvaluationContext): U[V]

  def grad[W[_], T](gc: GradientContext, v: ContinuousVariable[W, T])(implicit wOps: ContainerOps.Aux[W, T]): Gradient[W, U]

  def unary_-(): Expression[U, V, S] =
    UnaryExpression[U, V, S](this, new EvaluableValueFunc[V] {
      override def name: String = "neg"

      override def grad(gc: GradientContext, v: V) = {
        val valueVT = vt(gc).valueVT
        valueVT.negate(valueVT.one)
      }

      override def apply(ec: EvaluationContext, v: V) = {
        val vvt = vt(ec).valueVT
        vvt.negate(v)
      }
    })

  def toDouble(): Expression[U, Double, S] = {
    val self = this
    new Expression[U, Double, S] {

      override implicit val ops: Aux[U, Shape] = self.ops

      override implicit def vt: Evaluable[ValueOps[U, Double, S]] =
        self.vt.map { up ⇒
          ValueOps(Floating.doubleFloating, ops, up.shape)
        }

      override implicit val so = ScalarOps.liftBoth[U, Double, Double](ScalarOps.doubleOps, ops)

      override def apply(ec: EvaluationContext): U[Double] = {
        val valT = self.vt(ec).valueVT
        val up = self.apply(ec)
        ops.map(up)(valT.toDouble)
      }

      override def grad[W[_], T](gc: GradientContext, v: ContinuousVariable[W, T])(implicit wOps: Aux[W, T]): Gradient[W, U] = {
        self.grad(gc, v)
      }
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

      override def grad(gc: GradientContext, v: V) =
        sOps.times(vt(gc).valueVT.one, value)

      override def apply(ec: EvaluationContext, v: V) =
        sOps.times(v, value)
    })

  def /[W](value: W)(implicit sOps: ScalarOps[V, W]): Expression[U, V, S] =
    UnaryExpression[U, V, S](this, new EvaluableValueFunc[V] {
      val name = s"inv(${value})*"

      override def grad(gc: GradientContext, v: V) =
        sOps.div(vt(gc).valueVT.one, value)

      override def apply(ec: EvaluationContext, v: V) =
        sOps.div(v, value)
    })

  def +[W](rhs: W)(implicit sOps: ScalarOps[V, W]): Expression[U, V, S] = {
    UnaryExpression[U, V, S](this, new EvaluableValueFunc[V] {
      val name = s"${rhs}+"

      override def grad(gc: GradientContext, v: V) =
        vt(gc).valueVT.one

      override def apply(ec: EvaluationContext, v: V) =
        sOps.plus(v, rhs)
    })
  }

  def -[W](rhs: W)(implicit sOps: ScalarOps[V, W]): Expression[U, V, S] = {
    UnaryExpression[U, V, S](this, new EvaluableValueFunc[V] {
      val name = s"-${rhs}+"

      override def grad(gc: GradientContext, v: V) =
        vt(gc).valueVT.one

      override def apply(ec: EvaluationContext, v: V) =
        sOps.minus(v, rhs)
    })
  }
}
