package pp.ad


trait ValueOps[U[_], V] extends Floating[U[V]] {

  def valueVT: Floating[V]

}

object ValueOps {

  implicit def valueOps[U[_], V](implicit numV: Floating[V], ops: ContainerOps[U]): ValueOps[U, V] = new ValueOps[U, V] {

    override def valueVT = numV

    override def log = new UnaryValueFunc[U[V]] {

      private val upstream = numV.log

      override def grad(v: U[V]) = ops.map(v)(upstream.grad)

      override def apply(v: U[V]) = ops.map(v)(upstream.apply)
    }

    override def sum = new CollectValueFunc[U[V]] {

      private val upstream = numV.sum

      override def start = null.asInstanceOf[U[V]]

      override def apply(acc: U[V], v: U[V]) = ops.zipMap(acc, v) {
        (a, x) => upstream(if (a == null) upstream.start else a, x)
      }

      override def grad(a: U[V], v: U[V]) = ops.zipMap(a, v)(upstream)
    }

    override def div(x: U[V], y: U[V]) = ops.zipMap(x, y)(numV.div)

    override def plus(x: U[V], y: U[V]) = ops.zipMap(x, y)(numV.plus)

    override def minus(x: U[V], y: U[V]) = ops.zipMap(x, y)(numV.minus)

    override def times(x: U[V], y: U[V]) = ops.zipMap(x, y)(numV.times)

    override def negate(x: U[V]) = ops.map(x)(numV.negate)

    override def fromInt(x: Int) = ???

    override def toInt(x: U[V]) = ???

    override def toLong(x: U[V]) = ???

    override def toFloat(x: U[V]) = ???

    override def toDouble(x: U[V]) = ???

    override def compare(x: U[V], y: U[V]) = ???
  }
}
