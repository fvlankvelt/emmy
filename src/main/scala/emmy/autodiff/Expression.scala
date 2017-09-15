package emmy.autodiff

trait Node {

  def parents: Seq[Node] = Seq.empty
}

trait Expression[U[_], V, S] extends Node {

  type Shape = S

  implicit val vt: ValueOps[U, V, S]
  implicit val ops: ContainerOps.Aux[U, Shape]

  def shape: Shape

  def apply(ec: EvaluationContext[V]): U[V]

  def grad[W[_], T](gc: GradientContext[V], v: Variable[W, V, T])(implicit wOps: ContainerOps.Aux[W, T]): Gradient[W, U, V]

  def unary_-(): Expression[U, V, S] =
    UnaryExpression[U, V, S](this, new UnaryValueFunc[V] {
      override def name: String = "neg"

      override def grad(v: V) =
        vt.valueVT.negate(vt.valueVT.one)

      override def apply(v1: V) =
        vt.valueVT.negate(v1)
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
    UnaryExpression[U, V, S](this, new UnaryValueFunc[V] {
      val name = s"${value} *"

      override def grad(v: V) =
        sOps.times(vt.valueVT.one, value)

      override def apply(v1: V) =
        sOps.times(v1, value)
    })

  def /[W](value: W)(implicit sOps: ScalarOps[V, W]): Expression[U, V, S] =
    UnaryExpression[U, V, S](this, new UnaryValueFunc[V] {
      val name = s"inv(${value})*"

      override def grad(v: V) =
        sOps.div(vt.valueVT.one, value)

      override def apply(v1: V) =
        sOps.div(v1, value)
    })

  def +[W](rhs: W)(implicit sOps: ScalarOps[V, W]): Expression[U, V, S] = {
    UnaryExpression[U, V, S](this, new UnaryValueFunc[V] {
      val name = s"${rhs}+"

      override def grad(v: V) =
        vt.valueVT.one

      override def apply(v1: V) =
        sOps.plus(v1, rhs)
    })
  }

  def -[W](rhs: W)(implicit sOps: ScalarOps[V, W]): Expression[U, V, S] = {
    UnaryExpression[U, V, S](this, new UnaryValueFunc[V] {
      val name = s"-${rhs}+"

      override def grad(v: V) =
        vt.valueVT.one

      override def apply(v1: V) =
        sOps.minus(v1, rhs)
    })
  }
}
