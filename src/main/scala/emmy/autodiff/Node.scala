package emmy.autodiff


trait Node[U[_], V, S] {

  type Shape = S

  implicit val vt: ValueOps[U, V, S]
  implicit val ops: ContainerOps.Aux[U, Shape]

  def shape: Shape

  def apply(ec: EvaluationContext): U[V]

  def grad[W[_], T](gc: GradientContext, v: Variable[W, V, T])(implicit  wOps: ContainerOps.Aux[W, T]): Gradient[W, U, V]

  def unary_-(): Node[U, V, S] =
    UnaryNode[U, V, S](this, new UnaryValueFunc[V] {
      override def grad(v: V) =
        vt.valueVT.negate(v)

      override def apply(v1: V) =
        vt.valueVT.negate(v1)
    })

  def reciprocal(): Node[U, V, S] =
    Reciprocal(this)

  // element-wise ops

  def *(rhs: Node[U, V, S]): Node[U, V, S] =
    Multiply(this, rhs)

  def /(rhs: Node[U, V, S]): Node[U, V, S] =
    Multiply(this, rhs.reciprocal())

  def +(rhs: Node[U, V, S]): Node[U, V, S] =
    Add(this, rhs)

  def -(rhs: Node[U, V, S]): Node[U, V, S] =
    Add(this, -rhs)

  // scalar ops

  def *[W](value: W)(implicit sOps: ScalarOps[V, W]): Node[U, V, S] =
    UnaryNode[U, V, S](this, new UnaryValueFunc[V] {
      override def grad(v: V) =
        sOps.times(v, value)

      override def apply(v1: V) =
        sOps.times(v1, value)
    })

  def /[W](value: W)(implicit sOps: ScalarOps[V, W]): Node[U, V, S] =
    UnaryNode[U, V, S](this, new UnaryValueFunc[V] {
      override def grad(v: V) =
        sOps.div(v, value)

      override def apply(v1: V) =
        sOps.div(v1, value)
    })

  def +[W](rhs: W)(implicit sOps: ScalarOps[V, W]): Node[U, V, S] = {
    UnaryNode[U, V, S](this, new UnaryValueFunc[V] {
      override def grad(v: V) =
        vt.valueVT.one

      override def apply(v1: V) =
        sOps.plus(v1, rhs)
    })
  }

  def -[W](rhs: W)(implicit sOps: ScalarOps[V, W]): Node[U, V, S] = {
    UnaryNode[U, V, S](this, new UnaryValueFunc[V] {
      override def grad(v: V) =
        vt.valueVT.one

      override def apply(v1: V) =
        sOps.minus(v1, rhs)
    })
  }
}