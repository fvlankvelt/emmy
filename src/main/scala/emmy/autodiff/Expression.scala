package emmy.autodiff

trait Node {

  def parents: Seq[Node] = Seq.empty
}

trait Evaluable[+V] {
  self =>

  def apply(ec: EvaluationContext[_]): V

  def map[W](fn: V => W): Evaluable[W] = new Evaluable[W] {

    override def apply(ec: EvaluationContext[_]): W = {
      fn(self(ec))
    }

    override def toString() = {
      s"eval_map($self, $fn)"
    }
  }

}

object Evaluable {

  implicit def fromConstant[V](value: V): Evaluable[V] = new Evaluable[V] {

    override def apply(ec: EvaluationContext[_]): V = value

    override def toString() = {
      s"eval($value)"
    }
  }
}

trait Expression[U[_], V, S] extends Node {

  type Shape = S

  implicit val ops: ContainerOps.Aux[U, Shape]

  implicit def vt: Evaluable[ValueOps[U, V, S]]

  def apply(ec: EvaluationContext[V]): U[V]

  def grad[W[_], T](gc: GradientContext[V], v: Variable[W, V, T])(implicit wOps: ContainerOps.Aux[W, T]): Gradient[W, U, V]

  def unary_-(): Expression[U, V, S] =
    UnaryExpression[U, V, S](this, new EvaluableValueFunc[V] {
      override def name: String = "neg"

      override def grad(gc: GradientContext[_], v: V) = {
        val valueVT = vt(gc).valueVT
        valueVT.negate(valueVT.one)
      }

      override def apply(ec: EvaluationContext[_], v: V) = {
        val vvt = vt(ec).valueVT
        vvt.negate(v)
      }
    })

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

      override def grad(gc: GradientContext[_], v: V) =
        sOps.times(vt(gc).valueVT.one, value)

      override def apply(ec: EvaluationContext[_], v: V) =
        sOps.times(v, value)
    })

  def /[W](value: W)(implicit sOps: ScalarOps[V, W]): Expression[U, V, S] =
    UnaryExpression[U, V, S](this, new EvaluableValueFunc[V] {
      val name = s"inv(${value})*"

      override def grad(gc: GradientContext[_], v: V) =
        sOps.div(vt(gc).valueVT.one, value)

      override def apply(ec: EvaluationContext[_], v: V) =
        sOps.div(v, value)
    })

  def +[W](rhs: W)(implicit sOps: ScalarOps[V, W]): Expression[U, V, S] = {
    UnaryExpression[U, V, S](this, new EvaluableValueFunc[V] {
      val name = s"${rhs}+"

      override def grad(gc: GradientContext[_], v: V) =
        vt(gc).valueVT.one

      override def apply(ec: EvaluationContext[_], v: V) =
        sOps.plus(v, rhs)
    })
  }

  def -[W](rhs: W)(implicit sOps: ScalarOps[V, W]): Expression[U, V, S] = {
    UnaryExpression[U, V, S](this, new EvaluableValueFunc[V] {
      val name = s"-${rhs}+"

      override def grad(gc: GradientContext[_], v: V) =
        vt(gc).valueVT.one

      override def apply(ec: EvaluationContext[_], v: V) =
        sOps.minus(v, rhs)
    })
  }
}
