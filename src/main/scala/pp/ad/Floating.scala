package pp.ad

import scala.util.Random


trait Floating[V] extends Fractional[V] {

  def log: UnaryValueFunc[V]

  def lgamma: UnaryValueFunc[V]

  def sum: CollectValueFunc[V]

  def rnd: V

  def divS[Y](x: V, y: Y)(implicit so: ScalarOps[V, Y]): V = so.div(x, y)
}

object Floating {

  implicit val doubleFloating: Floating[Double] = new Floating[Double] {

    override def log = new UnaryValueFunc[Double] {

      def apply(x: Double) = scala.math.log(x)

      def grad(x: Double) = 1.0 / x
    }

    override def lgamma = new UnaryValueFunc[Double] {

      override def apply(value: Double) =
        breeze.numerics.lgamma(value)

      override def grad(value: Double) =
      breeze.numerics.digamma(value)
    }

    override def sum = new CollectValueFunc[Double] {

      override def apply(a: Double, b: Double) = a + b

      override val start = 0.0

      override def grad(a: Double, b: Double) = 1.0
    }

    override def rnd = Random.nextDouble()

    override def div(x: Double, y: Double) = x / y

    override def plus(x: Double, y: Double) = x + y

    override def minus(x: Double, y: Double) = x - y

    override def times(x: Double, y: Double) = x * y

    override def negate(x: Double) = -x

    override def fromInt(x: Int) = x.toDouble

    override def toInt(x: Double) = x.toInt

    override def toLong(x: Double) = x.toLong

    override def toFloat(x: Double) = x.toFloat

    override def toDouble(x: Double) = x

    override def compare(x: Double, y: Double) = x.compareTo(y)
  }

  implicit def toFloating[U[_], V, UVS](implicit numV: Floating[V], ops: ContainerOps.Aux[U, UVS]): Floating[U[V]] = ValueOps.valueOps[U, V, UVS]
}
