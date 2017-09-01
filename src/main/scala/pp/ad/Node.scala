package pp.ad


trait Node[U[_], V, S] extends (() => U[V]) {

  type Shape = S

  implicit val vt: ValueOps[U, V]
  implicit val ops: ContainerOps.Aux[U, Shape]
  protected lazy val _value: U[V] = value

  override final def apply() = _value

  def shape: Shape

  final def grad[W[_], T](v: Variable[W, V, T])(implicit wOps: ContainerOps.Aux[W, T]): Gradient[W, U, V] =
    v.get(this) {
      calcGrad(v)
    }

  def unary_-(): Node[U, V, S] = Scale(this, vt.valueVT.negate)

  def *(rhs: Node[U, V, S]): Node[U, V, S] = Multiply(this, rhs)

  def *[W](value: W)(implicit sOps: ScalarOps[V, W]): Node[U, V, S] = Scale[U, V, S](this, v => sOps.times(v, value))

  def /(rhs: Node[U, V, S]): Node[U, V, S] = Divide(this, rhs)

  def /[W](value: W)(implicit sOps: ScalarOps[V, W]): Node[U, V, S] = Scale[U, V, S](this, v => sOps.div(v, value))

  def +(rhs: Node[U, V, S]): Node[U, V, S] = Add(this, rhs)

  def -(rhs: Node[U, V, S]): Node[U, V, S] = Subtract(this, rhs)

  protected def value: U[V]

  protected def calcGrad[W[_], T](v: Variable[W, V, T])(implicit wOps: ContainerOps.Aux[W, T]): Gradient[W, U, V]
}