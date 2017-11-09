package emmy.autodiff

trait ScalarOps[V, Y] {

  def plus(v: V, y: Y): V

  def minus(v: V, y: Y): V

  def times(v: V, y: Y): V

  def div(v: V, y: Y): V
}

trait LowPriorityScalarOps {

  implicit def liftLeft[U[_], V, W](implicit base: ScalarOps[V, W], cOps: ContainerOps[U]): ScalarOps[U[V], W] =
    new ScalarOps[U[V], W] {

      override def plus(v: U[V], y: W) = cOps.map(v)(vi ⇒ base.plus(vi, y))

      override def minus(v: U[V], y: W) = cOps.map(v)(vi ⇒ base.minus(vi, y))

      override def times(v: U[V], y: W) = cOps.map(v)(vi ⇒ base.times(vi, y))

      override def div(v: U[V], y: W) = cOps.map(v)(vi ⇒ base.div(vi, y))

      override def toString = s"Lift left ($base) to $cOps"
    }

  implicit def liftBoth[U[_], V, Y](implicit base: ScalarOps[V, Y], ops: ContainerOps[U]): ScalarOps[U[V], U[Y]] =
    new ScalarOps[U[V], U[Y]] {

      override def plus(x: U[V], y: U[Y]) = ops.zipMap(x, y)(base.plus)

      override def minus(x: U[V], y: U[Y]) = ops.zipMap(x, y)(base.minus)

      override def times(x: U[V], y: U[Y]) = ops.zipMap(x, y)(base.times)

      override def div(x: U[V], y: U[Y]) = ops.zipMap(x, y)(base.div)

      override def toString = s"Lift ($base) to $ops"
    }

}

object ScalarOps extends LowPriorityScalarOps {

  implicit val doubleOps: ScalarOps[Double, Double] =
    new ScalarOps[Double, Double] {

      override def plus(v: Double, y: Double) = v + y

      override def minus(v: Double, y: Double) = v - y

      override def times(v: Double, y: Double) = v * y

      override def div(v: Double, y: Double) = v / y

      override def toString = "double ops"
    }

  implicit val intOps: ScalarOps[Int, Int] =
    new ScalarOps[Int, Int] {

      override def plus(v: Int, y: Int) = v + y

      override def minus(v: Int, y: Int) = v - y

      override def times(v: Int, y: Int) = v * y

      override def div(v: Int, y: Int) = v / y

      override def toString = "int ops"
    }

  implicit val intDoubleOps: ScalarOps[Double, Int] =
    new ScalarOps[Double, Int] {

      override def plus(v: Double, y: Int) = v + y

      override def minus(v: Double, y: Int) = v - y

      override def times(v: Double, y: Int) = v * y

      override def div(v: Double, y: Int) = v / y

      override def toString = "int2double ops"
    }

}
