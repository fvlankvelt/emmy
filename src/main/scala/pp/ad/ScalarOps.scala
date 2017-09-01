package pp.ad


trait ScalarOps[V, Y] {

  def times(v: V, y: Y): V

  def div(v: V, y: Y): V
}

object ScalarOps {

  implicit val doubleOps: ScalarOps[Double, Double] =
    new ScalarOps[Double, Double] {

      override def times(v: Double, y: Double) = v * y

      override def div(v: Double, y: Double) = v / y
    }

  implicit def liftOps[U[_], V, W](implicit base: ScalarOps[V, W], cOps: ContainerOps[U]): ScalarOps[U[V], W] =
    new ScalarOps[U[V], W] {

      override def times(v: U[V], y: W) = cOps.map(v)(vi => base.times(vi, y))

      override def div(v: U[V], y: W) = cOps.map(v)(vi => base.div(vi, y))
    }
}
