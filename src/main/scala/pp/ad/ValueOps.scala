package pp.ad


trait ValueOps[U[_], V, Shape] extends Floating[U[V]] {

  def valueVT: Floating[V]

  def ops: ContainerOps.Aux[U, Shape]

  def bind(shape: Shape) =
    UnaryValueOps[U, V, Shape](shape)(valueVT, ops)
}

object ValueOps {

//  type Aux[U[_], V, S] = ValueOps[U, V] { type Shape = S }

  implicit def valueOps[U[_], V, UVS](implicit numV: Floating[V], cOps: ContainerOps.Aux[U, UVS]): ValueOps[U, V, UVS] = new BinaryValueOps[U, V, UVS] {

    override def valueVT = numV

    override implicit def ops = cOps
  }
}

trait BinaryValueOps[U[_], V, S] extends ValueOps[U, V, S] {

  override def log = new UnaryValueFunc[U[V]] {

    private val upstream = valueVT.log

    override def grad(v: U[V]) = ops.map(v)(upstream.grad)

    override def apply(v: U[V]) = ops.map(v)(upstream.apply)
  }

  override def lgamma = new UnaryValueFunc[U[V]] {

    private val upstream = valueVT.lgamma

    override def grad(v: U[V]) = ops.map(v)(upstream.grad)

    override def apply(v: U[V]) = ops.map(v)(upstream.apply)
  }

  override def sum = new CollectValueFunc[U[V]] {

    private val upstream = valueVT.sum

    override def start = null.asInstanceOf[U[V]]

    override def apply(acc: U[V], v: U[V]) = ops.zipMap(acc, v) {
      (a, x) => upstream(if (a == null) upstream.start else a, x)
    }

    override def grad(a: U[V], v: U[V]) = ops.zipMap(a, v)(upstream)
  }

  override def div(x: U[V], y: U[V]) = ops.zipMap(x, y)(valueVT.div)

  override def plus(x: U[V], y: U[V]) = ops.zipMap(x, y)(valueVT.plus)

  override def minus(x: U[V], y: U[V]) = ops.zipMap(x, y)(valueVT.minus)

  override def times(x: U[V], y: U[V]) = ops.zipMap(x, y)(valueVT.times)

  override def negate(x: U[V]) = ops.map(x)(valueVT.negate)

  override def rnd: U[V] = invalidOp("rnd")

  override def fromInt(x: Int): U[V] = invalidOp("fromInt")

  override def toInt(x: U[V]) = invalidOp("toInt")

  override def toLong(x: U[V]) = invalidOp("toLong")

  override def toFloat(x: U[V]) = invalidOp("toFloat")

  override def toDouble(x: U[V]) = invalidOp("toDouble")

  override def compare(x: U[V], y: U[V]) = invalidOp("compare")

  private def invalidOp[R](op: String): R = {
    throw new UnsupportedOperationException(s"Operation '$op' is not valid on unbound ValueOps")
  }
}

case class UnaryValueOps[U[_], V, S](shape: S)(implicit
                                               val valueVT: Floating[V],
                                               val ops: ContainerOps.Aux[U, S])
  extends BinaryValueOps[U, V, S] {

  override def fromInt(x: Int) = ops.fill(shape, valueVT.fromInt(x))

  override def rnd = ops.fill(shape, valueVT.rnd)
}

