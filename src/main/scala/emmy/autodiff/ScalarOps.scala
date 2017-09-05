package emmy.autodiff


trait ScalarOps[V, Y] {

  def plus(v: V, y: Y): V

  def minus(v: V, y: Y): V

  def times(v: V, y: Y): V

  def div(v: V, y: Y): V
}

object ScalarOps {

  implicit val doubleOps: ScalarOps[Double, Double] =
    new ScalarOps[Double, Double] {

      override def plus(v: Double, y: Double) = v + y

      override def minus(v: Double, y: Double) = v - y

      override def times(v: Double, y: Double) = v * y

      override def div(v: Double, y: Double) = v / y
    }

  implicit def liftOps[U[_], V, W](implicit base: ScalarOps[V, W], cOps: ContainerOps[U]): ScalarOps[U[V], W] =
    new ScalarOps[U[V], W] {

      override def plus(v: U[V], y: W) = cOps.map(v)(vi => base.plus(vi, y))

      override def minus(v: U[V], y: W) = cOps.map(v)(vi => base.minus(vi, y))

      override def times(v: U[V], y: W) = cOps.map(v)(vi => base.times(vi, y))

      override def div(v: U[V], y: W) = cOps.map(v)(vi => base.div(vi, y))
    }
}
